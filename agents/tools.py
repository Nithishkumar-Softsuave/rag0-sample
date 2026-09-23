"""The two tools both the agent and the fixed workflow are built from.

Deliberately deterministic, plain-Python, no LLM call -- a tool needs to be
testable and trustworthy on its own before an agent is trusted to call it.
Parses `news_articles/*.txt` directly rather than going through
`rag_chat.retrieval.retrieve()`: that pipeline returns unstructured chunk
text meant for a chat model to read, not the structured {shop, item, price,
currency} records these tools need to compare across currencies.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache

from rag_chat.config import get_settings

_SERVING_SIZE_NOTE = re.compile(r"\s*\(\d+\s*pcs?\)$", re.IGNORECASE)

SHOP_LINE = re.compile(r"^(?P<shop>.+?)\s*-\s*.+Menu\s*$", re.MULTILINE)
CURRENCY_LINE = re.compile(r"Currency:.*\(([A-Z]{3})\)")
ITEM_LINE = re.compile(r"^\*\s*(?P<item>.+?)\s*\|\s*Price:\s*(?:Rs\.?\s*|\$)?(?P<amount>[\d.]+)", re.MULTILINE)

# A fixed, offline exchange rate table -- there is no live FX API here, and
# the corpus only ever mixes these two currencies (see docs/week7-agents.md).
EXCHANGE_RATES = {
    ("SGD", "INR"): 61.0,
    ("INR", "SGD"): 1 / 61.0,
}


@dataclass(frozen=True)
class MenuRecord:
    shop: str
    item: str
    price: float
    currency: str


@lru_cache(maxsize=1)
def load_catalog() -> tuple[MenuRecord, ...]:
    """Parse every menu file once into a flat, structured catalog."""
    directory = get_settings().articles_dir
    records: list[MenuRecord] = []
    for path in sorted(directory.glob("*.txt")):
        text = path.read_text(encoding="utf-8")
        shop_match = SHOP_LINE.search(text)
        currency_match = CURRENCY_LINE.search(text)
        if not shop_match or not currency_match:
            continue
        shop, currency = shop_match.group("shop"), currency_match.group(1)
        for item_match in ITEM_LINE.finditer(text):
            records.append(MenuRecord(
                shop=shop,
                item=item_match.group("item"),
                price=float(item_match.group("amount")),
                currency=currency,
            ))
    return tuple(records)


def search_menus(item: str) -> list[dict]:
    """Every menu record whose item name matches `item`, either direction.

    Matching both ways ("Filter Coffee" in the item, or the item's exact
    name inside a longer search phrase) is what lets a loosely-worded tool
    call still find the right rows.
    """
    needle = item.strip().lower()
    if not needle:
        return []
    matches = [
        record for record in load_catalog()
        if needle in record.item.lower() or record.item.lower() in needle
    ]
    return [
        {"shop": record.shop, "item": record.item, "price": record.price, "currency": record.currency}
        for record in matches
    ]


def core_item_name(item: str) -> str:
    """Strip a trailing serving-size note, e.g. "Medu Vada (2 pcs)" -> "Medu Vada".

    Deliberately narrow: only a "(<number> pcs)" pattern is stripped, not any
    parenthetical. An earlier, broader version stripped "Meals (Veg Thali)"
    down to the bare word "Meals" -- which then falsely matched inside
    "Chettinad Chicken Meals" and any other item ending in "Meals". A
    parenthetical is usually a meaningful qualifier (which thali, which tea
    variant); only the serving-size count is genuinely safe to drop.
    """
    return _SERVING_SIZE_NOTE.sub("", item).strip()


def search_menus_for_agent(item: str) -> dict:
    """search_menus(), plus a warning when the matches are different dishes.

    Week 8 fix for the outcome-vs-trajectory gap in
    docs/week8-agent-failures.md: substring matching makes "Thali" return
    "Kaveri Special Thali" *and* "Meals (Veg Thali)" -- different products --
    and the agent used to compare them as one item and name a "cheapest".
    The tool now says so explicitly, instead of relying on the model to
    notice. The plain search_menus() is unchanged, so the fixed workflow and
    its tests are unaffected.
    """
    matches = search_menus(item)
    result: dict = {"matches": matches}
    if not matches:
        result["note"] = f"No menu item matches '{item}'. Tell the user it was not found; do not guess a price."
        return result
    distinct = sorted({core_item_name(match["item"]) for match in matches})
    if len(distinct) > 1:
        result["warning"] = (
            f"'{item}' matched {len(distinct)} DIFFERENT dishes: {distinct}. These are not the "
            "same item. Do not name one of them the 'cheapest <item>' as if they were comparable -- "
            "list each dish separately by its full name, and only compare prices between records "
            "with the same dish name."
        )
    return result


def convert_currency(amount: float, from_currency: str, to_currency: str) -> dict:
    """Convert `amount` between currencies using the fixed rate table above."""
    from_currency, to_currency = from_currency.upper(), to_currency.upper()
    if from_currency == to_currency:
        return {"amount": amount, "currency": to_currency}
    rate = EXCHANGE_RATES.get((from_currency, to_currency))
    if rate is None:
        return {"error": f"No exchange rate available for {from_currency} -> {to_currency}."}
    return {"amount": round(amount * rate, 2), "currency": to_currency}


TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "search_menus",
            "description": (
                "Search every restaurant's menu for a food or drink item by name. "
                "Returns {matches: [{shop, item, price, currency}]}, plus a `warning` when the "
                "matches are different dishes, or a `note` when nothing matched."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "item": {"type": "string", "description": "The food or drink item name to search for, e.g. 'Filter Coffee'."},
                },
                "required": ["item"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "convert_currency",
            "description": "Convert an amount from one currency to another. Supported currencies: INR, SGD.",
            "parameters": {
                "type": "object",
                "properties": {
                    "amount": {"type": "number"},
                    "from_currency": {"type": "string", "description": "3-letter currency code, e.g. SGD."},
                    "to_currency": {"type": "string", "description": "3-letter currency code, e.g. INR."},
                },
                "required": ["amount", "from_currency", "to_currency"],
            },
        },
    },
]

TOOL_FUNCTIONS = {
    "search_menus": lambda args: search_menus_for_agent(args["item"]),
    "convert_currency": lambda args: convert_currency(args["amount"], args["from_currency"], args["to_currency"]),
}
