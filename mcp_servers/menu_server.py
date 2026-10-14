"""MCP server exposing this app's real capability: menu search + price conversion.

This is the "build your own server that another person's agent can call"
half of Week 9. The tool logic is not rewritten -- it is agents/tools.py,
the same deterministic, tested code the Week 7/8 agent used. MCP only
changes how it is *offered*: instead of a Python dict the agent imports,
it is a server any MCP host can discover and call.

    python -m mcp_servers.menu_server                      # stdio (local host launches it)
    python -m mcp_servers.menu_server --transport http     # remote, needs MENU_MCP_TOKENS

Offers all three MCP primitives:
- tools      search_menus, price_dishes, convert_currency  (the model decides to call these)
- resource   menu://shops                   (read-only data the host/user can pull in)
- prompt     compare_item_prices            (a reusable prompt template a user picks)

Arguments are validated here, on the server, from the type hints -- a
server cannot assume every caller runs our host's guards (agents/guards.py).
"""
from __future__ import annotations

import json
from typing import Annotated, Literal

from pydantic import BaseModel, Field

from agents.tools import convert_currency as _convert_currency
from agents.tools import load_catalog, search_menus_for_agent
from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ToolAnnotations
from mcp_servers import _serving

Currency = Literal["INR", "SGD"]

INSTRUCTIONS = (
    "Menu price tools for a set of South Indian and Singapore restaurants. "
    "Call search_menus for every item the question mentions before answering. "
    "If prices are in different currencies, convert to a common one with "
    "convert_currency before comparing -- only when currencies actually differ, "
    "and only on a price a search returned. Name the shop for every price. "
    "If a search result carries a `warning` that the matches are different "
    "dishes, list each dish by its full name and never call one of them the "
    "cheapest across different dishes. If a search has a `note` that nothing "
    "matched, say it was not found instead of guessing. To price several "
    "dishes -- for example every alternative in a swap -- call "
    "price_dishes once with all of them, not search_menus once per dish."
)

# Every tool here only reads a fixed local catalog: no side effects, same
# answer every time, no outside world. Hosts use these hints to decide what
# can run without asking the user.
READ_ONLY = ToolAnnotations(readOnlyHint=True, destructiveHint=False, idempotentHint=True, openWorldHint=False)


class MenuMatch(BaseModel):
    shop: str
    item: str
    price: float
    currency: Currency


class MenuSearchResult(BaseModel):
    matches: list[MenuMatch]
    warning: str | None = Field(None, description="Set when the matches are DIFFERENT dishes that must not be compared as one item.")
    note: str | None = Field(None, description="Set when nothing matched.")


class ItemSearchResult(MenuSearchResult):
    searched: str = Field(description="The item this result is for.")


class BulkSearchResult(BaseModel):
    results: list[ItemSearchResult]


class Conversion(BaseModel):
    amount: float
    currency: Currency


class Shop(BaseModel):
    shop: str
    currency: Currency
    items: int


def search_menus(
    item: Annotated[str, Field(min_length=1, max_length=100, description="Food or drink item to search for, e.g. 'Filter Coffee'.")],
) -> MenuSearchResult:
    """Search every restaurant's menu for a food or drink item by name.

    Returns each match's shop, full item name, price and currency, plus a
    `warning` when the matches are different dishes, or a `note` when
    nothing matched.
    """
    if not item.strip():
        raise ToolError("item must not be blank -- pass the dish or drink name to search for.")
    return MenuSearchResult.model_validate(search_menus_for_agent(item))


MAX_BULK_ITEMS = 10


def price_dishes(
    items: Annotated[
        list[Annotated[str, Field(min_length=1, max_length=100)]],
        Field(min_length=1, max_length=MAX_BULK_ITEMS, description="Items to search for, e.g. ['Plain Idli', 'Medu Vada']."),
    ],
) -> BulkSearchResult:
    """Search every restaurant's menu for several items in one call.

    Same result as search_menus for each item, in order. Week 11: the live
    logs showed swap questions pricing alternatives one search_menus call at
    a time -- each call re-sends the whole conversation to the model, so a
    five-dish swap cost ~5x a normal answer and ran out of steps before
    answering. One call for all of them fixes both.

    It was first named search_menus_many. In the first after-fix run, 2 of
    6 requests then crashed on a garbled tool name,
    'search_menus<|channel|>commentary' (never seen in the 24 baseline
    requests). The likely cause is two tool names sharing the
    'search_menus' prefix, so it got a name no other tool starts with.
    """
    seen: list[str] = []
    for item in items:
        if not item.strip():
            raise ToolError("items must not contain a blank name.")
        if item.strip().lower() not in (s.lower() for s in seen):
            seen.append(item.strip())
    return BulkSearchResult(results=[
        ItemSearchResult(searched=item, **search_menus_for_agent(item)) for item in seen
    ])


def convert_currency(
    amount: Annotated[float, Field(gt=0, le=100_000, description="The price to convert.")],
    from_currency: Annotated[Currency, Field(description="Currency the amount is in.")],
    to_currency: Annotated[Currency, Field(description="Currency to convert into.")],
) -> Conversion:
    """Convert a menu price between INR and SGD using a fixed exchange rate."""
    result = _convert_currency(amount, from_currency, to_currency)
    if "error" in result:
        raise ToolError(result["error"])
    return Conversion.model_validate(result)


def list_shops() -> list[Shop]:
    """Every shop in the catalog, its currency, and how many items it lists."""
    shops: dict[str, Shop] = {}
    for record in load_catalog():
        shop = shops.setdefault(record.shop, Shop(shop=record.shop, currency=record.currency, items=0))
        shop.items += 1
    return list(shops.values())


def shops_resource() -> str:
    """Body of the menu://shops resource: list_shops() as JSON text."""
    return json.dumps([shop.model_dump() for shop in list_shops()], indent=2)


def compare_item_prices(item: str, currency: Currency = "INR") -> str:
    """Prompt template: compare one item's price across every shop."""
    return (
        f"Find every shop that sells {item} and compare their prices in {currency}. "
        "List each shop with its price, then say which is cheapest."
    )


def build_server(**auth) -> MCPServer:
    """A fresh server instance. `auth` is token_verifier/auth for HTTP mode (see _serving.py)."""
    server = MCPServer(
        "menu",
        title="Restaurant Menu Prices",
        version="1.0.0",
        instructions=INSTRUCTIONS,
        log_level=_serving.LOG_LEVEL,
        **auth,
    )
    server.tool(title="Search menus", annotations=READ_ONLY)(search_menus)
    server.tool(title="Search menus for several items", annotations=READ_ONLY)(price_dishes)
    server.tool(title="Convert currency", annotations=READ_ONLY)(convert_currency)
    server.resource("menu://shops", name="shops", title="Shops in the catalog", mime_type="application/json")(
        shops_resource
    )
    server.prompt(title="Compare an item's price across shops")(compare_item_prices)
    return server


def main() -> None:
    _serving.run(
        build_server,
        description="Restaurant menu MCP server",
        default_port=8765,
        token_env="MENU_MCP_TOKENS",
        scope="menu:read",
    )


if __name__ == "__main__":
    main()
