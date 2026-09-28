"""Track B bolt-on: the ingredient database, as its own MCP server.

The Week 9 test for Track B is "bolt on the ingredient database server
without touching the agent". So this server is deliberately standalone: it
imports nothing from agents/ or rag_chat/, owns its own data file
(mcp_servers/data/ingredients.json), and could be run by a different team
on a different machine. Adding it to the agent is one entry in
mcp_config.json -- no agent code changes. See docs/week9-mcp.md.

    python -m mcp_servers.ingredient_server                    # stdio
    python -m mcp_servers.ingredient_server --transport http   # remote, needs INGREDIENT_MCP_TOKENS

Tools:
- lookup_ingredients(dish)                    what is in a dish, allergens, diet
- find_dishes(free_from, diet, contains)      filter dishes by allergen / diet / ingredient
Resource:
- ingredients://allergens                     the allergen names find_dishes accepts
"""
from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, Field

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ToolAnnotations
from mcp_servers import _serving

DATA_FILE = Path(__file__).resolve().parent / "data" / "ingredients.json"

Diet = Literal["vegan", "vegetarian", "pescatarian", "non-vegetarian"]
Allergen = Literal[
    "gluten", "dairy", "egg", "tree nuts", "peanuts", "soy", "fish",
    "shellfish", "sesame", "mustard", "sulphites", "alcohol",
]

# Each diet also satisfies the looser ones after it: a vegan dish is fine
# for someone asking for vegetarian, a vegetarian one for pescatarian.
DIET_SATISFIES: dict[str, set[str]] = {
    "vegan": {"vegan"},
    "vegetarian": {"vegan", "vegetarian"},
    "pescatarian": {"vegan", "vegetarian", "pescatarian"},
    "non-vegetarian": {"vegan", "vegetarian", "pescatarian", "non-vegetarian"},
}

RECIPE_CAVEAT = (
    "Standard recipe, not a specific shop's: a restaurant's own recipe can differ. "
    "For a severe allergy, the customer should confirm with the restaurant."
)

INSTRUCTIONS = (
    "Ingredient database for the same restaurant dishes. Use lookup_ingredients "
    "for what a dish contains, its allergens and whether it is vegan/vegetarian; "
    "use find_dishes to list dishes free from given allergens or matching a diet. "
    "Pass a dish's name without any serving-size note like '(2 pcs)'. "
    "Whenever you state an allergen or diet fact, add that it is the standard "
    "recipe and the restaurant should confirm for a severe allergy."
)

READ_ONLY = ToolAnnotations(readOnlyHint=True, destructiveHint=False, idempotentHint=True, openWorldHint=False)

_SERVING_SIZE_NOTE = re.compile(r"\s*\(\d+\s*pcs?\)\s*$", re.IGNORECASE)


class Dish(BaseModel):
    dish: str
    aliases: list[str] = []
    diet: Diet
    ingredients: list[str]
    served_with: list[str] = []
    allergens: list[Allergen]
    spice_level: Literal["none", "mild", "medium", "hot"]


class IngredientLookup(BaseModel):
    matches: list[Dish]
    caveat: str = RECIPE_CAVEAT


class DishSummary(BaseModel):
    dish: str
    diet: Diet
    allergens: list[Allergen]


class DishFilterResult(BaseModel):
    dishes: list[DishSummary]
    caveat: str = RECIPE_CAVEAT


@lru_cache(maxsize=1)
def load_dishes() -> tuple[Dish, ...]:
    """Parse and validate the database once; a malformed entry fails at startup, not mid-answer."""
    raw = json.loads(DATA_FILE.read_text(encoding="utf-8"))
    return tuple(Dish.model_validate(entry) for entry in raw["dishes"])


def _names(dish: Dish) -> list[str]:
    return [dish.dish.lower(), *(alias.lower() for alias in dish.aliases)]


def match_dishes(query: str) -> list[Dish]:
    """Exact name/alias matches if any; otherwise substring matches either way.

    Exact-first stops "Rava Dosa" also returning "Masala Dosa" (whose alias
    is plain "dosa"), while a loose query like "thali" still finds both
    thalis. The serving-size note menus add, "(2 pcs)", is ignored.
    """
    needle = _SERVING_SIZE_NOTE.sub("", query).strip().lower()
    if not needle:
        return []
    exact = [dish for dish in load_dishes() if needle in _names(dish)]
    if exact:
        return exact
    return [
        dish for dish in load_dishes()
        if any(needle in name or name in needle for name in _names(dish))
    ]


def lookup_ingredients(
    dish: Annotated[str, Field(min_length=1, max_length=100, description="Dish or drink name, e.g. 'Masala Dosa'.")],
) -> IngredientLookup:
    """Look up a dish's ingredients, allergens, diet (vegan/vegetarian/...) and spice level."""
    matches = match_dishes(dish)
    if not matches:
        # A recoverable error: the model reads this and can retry with a
        # listed name instead of giving up or inventing ingredients.
        known = ", ".join(sorted(d.dish for d in load_dishes()))
        raise ToolError(f"No dish named '{dish}' in the ingredient database. Known dishes: {known}.")
    return IngredientLookup(matches=matches)


def find_dishes(
    free_from: Annotated[list[Allergen] | None, Field(description="Allergens the dish must NOT contain.")] = None,
    diet: Annotated[Diet | None, Field(description="Only dishes suitable for this diet.")] = None,
    contains: Annotated[str | None, Field(max_length=60, description="Only dishes with this ingredient, e.g. 'coconut'.")] = None,
) -> DishFilterResult:
    """List dishes that are free from given allergens, suit a diet, and/or contain an ingredient."""
    if not free_from and not diet and not contains:
        raise ToolError("Give at least one filter: free_from, diet, or contains.")
    avoid = set(free_from or [])
    ingredient = (contains or "").strip().lower()
    dishes = [
        DishSummary(dish=dish.dish, diet=dish.diet, allergens=dish.allergens)
        for dish in load_dishes()
        if not avoid & set(dish.allergens)
        and (diet is None or dish.diet in DIET_SATISFIES[diet])
        and (not ingredient or any(ingredient in part.lower() for part in dish.ingredients))
    ]
    return DishFilterResult(dishes=dishes)


def allergens_resource() -> str:
    """Body of ingredients://allergens: the allergen names find_dishes accepts."""
    raw = json.loads(DATA_FILE.read_text(encoding="utf-8"))
    return json.dumps(raw["allergen_vocabulary"])


def build_server(**auth) -> MCPServer:
    """A fresh server instance. `auth` is token_verifier/auth for HTTP mode (see _serving.py)."""
    load_dishes()  # validate the data file before accepting any connection
    server = MCPServer(
        "ingredients",
        title="Dish Ingredient Database",
        version="1.0.0",
        instructions=INSTRUCTIONS,
        log_level=_serving.LOG_LEVEL,
        **auth,
    )
    server.tool(title="Look up a dish's ingredients", annotations=READ_ONLY)(lookup_ingredients)
    server.tool(title="Find dishes by allergen or diet", annotations=READ_ONLY)(find_dishes)
    server.resource("ingredients://allergens", name="allergens", title="Allergen vocabulary", mime_type="application/json")(
        allergens_resource
    )
    return server


def main() -> None:
    _serving.run(
        build_server,
        description="Dish ingredient database MCP server",
        default_port=8766,
        token_env="INGREDIENT_MCP_TOKENS",
        scope="ingredients:read",
    )


if __name__ == "__main__":
    main()
