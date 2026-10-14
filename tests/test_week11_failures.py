"""Week 11: production failures turned into permanent tests -- no API key needed.

1. The support drill's bad answer ("the dairy-free swap that wasn't"):
   the agent called Ven Pongal dairy-free without looking it up. It has ghee.
   Now every free-from claim must be grounded in this run's ingredient
   results (agents/allergen_claims.py, rule 4 of output_validation.py).
2. The live-traffic failure: swap questions priced alternatives one
   search_menus call at a time and hit the 6-step limit -- the customer got
   "Stopped: exceeded step limit". The menu server now offers
   price_dishes (tests/test_mcp.py), and the host's prompt tells the
   model to use it.
"""
from datetime import datetime, timezone

import pytest

from agents.agent import AgentStep
from agents.allergen_claims import check_claims, free_from_claims, name_index
from agents.output_validation import validate_answer
from mcp_servers import menu_server
from mcp_servers.ingredient_server import find_dishes, load_dishes, lookup_ingredients
from observability import drill
from observability.logs import audit, recheck
from tests.test_mcp import MENU, host_for

INDEX = name_index({d.dish: d.aliases for d in load_dishes()})
DATABASE = {d.dish: set(d.allergens) for d in load_dishes()}


def _false_dairy_claims(answer: str) -> list[str]:
    return sorted(c.dish for c in free_from_claims(answer, INDEX) if c.allergen == "dairy" and "dairy" in DATABASE[c.dish])


def _dairy_claims(answer: str) -> list[str]:
    return sorted(c.dish for c in free_from_claims(answer, INDEX) if c.allergen == "dairy")


# -- 1. the planted bad answer --------------------------------------------

def test_the_planted_answer_is_now_rejected() -> None:
    """Replay of the drill trace: Masala Dosa looked up, Ven Pongal only priced, then called dairy-free."""
    steps = [
        AgentStep("lookup_ingredients", {"dish": "Masala Dosa"}, lookup_ingredients("Masala Dosa").model_dump()),
        AgentStep("search_menus", {"item": "Ven Pongal"}, menu_server.search_menus("Ven Pongal").model_dump()),
    ]
    issues = validate_answer(drill.PLANTED_ANSWER, steps, "")
    assert issues == ["says Ven Pongal is dairy-free without looking up its ingredients in this run"]


def test_a_claim_the_lookup_contradicts_is_rejected() -> None:
    steps = [AgentStep("lookup_ingredients", {"dish": "Ven Pongal"}, lookup_ingredients("Ven Pongal").model_dump())]
    issues = validate_answer("Ven Pongal is dairy-free.", steps, "")
    assert issues == ["says Ven Pongal is dairy-free, but its ingredient lookup lists dairy"]


def test_a_grounded_swap_passes() -> None:
    steps = [
        AgentStep("lookup_ingredients", {"dish": "Masala Dosa"}, lookup_ingredients("Masala Dosa").model_dump()),
        AgentStep("find_dishes", {"free_from": ["dairy"]}, find_dishes(free_from=["dairy"]).model_dump()),
        AgentStep("search_menus", {"item": "Rava Dosa"}, menu_server.search_menus("Rava Dosa").model_dump()),
    ]
    answer = ("Masala Dosa isn't dairy-free (it has ghee). Swap it for Rava Dosa at Vaigai Tiffin Center "
              "(Rs. 60), which is dairy-free -- note it contains gluten.")
    assert validate_answer(answer, steps, "") == []


def test_audit_finds_exactly_the_planted_answer_in_the_drill_log() -> None:
    traces = drill.build(datetime(2026, 10, 12, 12, tzinfo=timezone.utc), count=400)
    [finding] = audit(traces, "dairy")
    assert finding["user_id"] == drill.PLANTED_USER and finding["dish"] == "Ven Pongal" and not finding["looked_up"]
    [(trace, issues)] = recheck(traces)  # the new check rejects it and nothing else
    assert trace["user_id"] == drill.PLANTED_USER and "Ven Pongal" in issues[0]


# -- the claim parser: what counts as "X is dairy-free" -------------------

@pytest.mark.parametrize("answer, claimed", [
    ("Ven Pongal is dairy-free.", ["Ven Pongal"]),
    ("Ven Pongal is dairy‑free.", ["Ven Pongal"]),                      # the model's non-breaking hyphen
    ("Swap it for the Ven Pongal (Rs. 55) - a dairy-free rice dish.", ["Ven Pongal"]),  # "Rs." is not a sentence end
    ("Ven Pongal has no dairy.", ["Ven Pongal"]),
    ("Ven Pongal is vegan.", ["Ven Pongal"]),                                # vegan implies no dairy
    ("Masala Dosa contains ghee, so it is not dairy-free.", []),
    ("Ven Pongal isn't vegan.", []),
    ("There is no dairy-free version of Curd Rice.", []),
    ("Here are dairy-free dishes instead of Masala Dosa:\n\n- Plain Idli\n- Medu Vada", ["Medu Vada", "Plain Idli"]),
    ("Swap Masala Dosa for Rava Dosa, which is dairy-free.", ["Rava Dosa"]),  # the swapped-out dish isn't claimed
    ("| Dish (dairy-free) | Price |\n|---|---|\n| Plain Idli | 30 |\n| Ven Pongal | 55 |", ["Plain Idli", "Ven Pongal"]),
    ("Masala Dosa is Rs. 70 at Sri Kaveri Bhavan.", []),
    # Live traffic, req_9f309f4934d9: a correct answer with a "Dairy-free?" column -- not a claim.
    ("| Dish | Shop | Price | Dairy‑free? |\n|---|---|---|---|\n| Masala Dosa | Sri Kaveri Bhavan | 70 | No |", []),
])
def test_free_from_claims(answer: str, claimed: list[str]) -> None:
    assert _dairy_claims(answer) == claimed


def test_live_long_black_answer_is_now_rejected() -> None:
    """Live traffic, req_96d4919d6664: the lookup listed dairy, the answer said the opposite --
    by the alias "Long Black", which the first version of the check didn't know."""
    steps = [AgentStep("lookup_ingredients", {"dish": "Long Black coffee"}, lookup_ingredients("Long Black").model_dump())]
    answer = ("The Long Black coffee is listed as a vegetarian beverage that contains only espresso and no milk. "
              "Its allergen information does not include dairy, so it is considered dairy‑free.")
    issues = validate_answer(answer, steps, "")
    assert issues == ["says Freshly Brewed Coffee (Latte / Cappuccino / Long Black) is dairy-free, "
                      "but its ingredient lookup lists dairy"]


def test_only_dishes_this_run_returned_are_checked() -> None:
    """With no tool results there is nothing to name a dish from -- the check can't fire on chit-chat."""
    assert check_claims("Ven Pongal is dairy-free.", []) == ([], [])


# -- 2. swap questions ran out of steps -------------------------------------

def test_host_prompt_tells_the_model_to_price_swaps_in_one_call() -> None:
    with host_for(MENU, menu=menu_server.build_server()) as host:
        assert "price_dishes" in host.tool_functions
        assert "call price_dishes once with all of them" in host.system_prompt


# -- 3. a garbled tool call crashed the request -----------------------------

def _groq_400(code: str):
    import httpx
    from openai import BadRequestError
    response = httpx.Response(400, request=httpx.Request("POST", "https://example.invalid"))
    body = {"error": {"message": "Tool call validation failed", "type": "invalid_request_error", "code": code}}
    return BadRequestError("400", response=response, body=body)


def test_garbled_tool_call_is_retried_and_counted() -> None:
    """Live trace req_8e279f00d304: 'search_menus<|channel|>commentary' -> Groq 400 -> request crashed."""
    from types import SimpleNamespace
    from unittest.mock import patch

    import rag_chat.client as client_module
    from rag_chat.client import create_chat_completion, usage_scope

    ok = SimpleNamespace(choices=[], usage=None)
    calls = iter([_groq_400("tool_use_failed"), ok])

    def create(**_):
        item = next(calls)
        if isinstance(item, Exception):
            raise item
        return item

    fake = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    with patch.object(client_module, "get_chat_client", return_value=fake), usage_scope() as usage:
        assert create_chat_completion(model="m", messages=[]) is ok
    assert usage.malformed_tool_call_retries == 1


def test_other_bad_requests_and_repeat_garbling_still_raise() -> None:
    from types import SimpleNamespace
    from unittest.mock import patch

    from openai import BadRequestError

    import rag_chat.client as client_module
    from rag_chat.client import create_chat_completion

    for errors in ([_groq_400("context_length_exceeded")], [_groq_400("tool_use_failed")] * 3):
        calls = iter(errors)

        def create(**_):
            raise next(calls)

        fake = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        with patch.object(client_module, "get_chat_client", return_value=fake), pytest.raises(BadRequestError):
            create_chat_completion(model="m", messages=[])


def test_one_rows_vegan_cell_does_not_leak_into_the_rows_below() -> None:
    """Live trace req_59134b5493b1: a correct dairy-free table was rejected for calling Kari Dosai egg-free,
    because Plain Idli's "Vegan" cell became a header for every later row. The retry then ran out the clock."""
    answer = ("Here are some dishes that are **free from dairy** and can be enjoyed instead of a Masala Dosa:\n\n"
              "| Dish | Typical Diet | Notable Allergens |\n|------|------|------|\n"
              "| Plain Idli | Vegan | None |\n| Rava Dosa | Vegan | Gluten |\n"
              "| Kari Dosai (egg\u2011based) | Non\u2011vegetarian | Egg |\n"
              "| Chettinad Chicken Meals | Non\u2011vegetarian | Mustard |\n\n"
              "All of the items above are listed as dairy\u2011free in the menu data.")
    claims = {(c.dish, c.allergen) for c in free_from_claims(answer, INDEX)}
    assert ("Kari Dosai", "dairy") in claims and ("Plain Idli", "egg") in claims  # row's own "Vegan"
    assert ("Kari Dosai", "egg") not in claims and ("Chettinad Chicken Meals", "egg") not in claims
    steps = [AgentStep("find_dishes", {"free_from": ["dairy"]}, find_dishes(free_from=["dairy"]).model_dump())]
    assert validate_answer(answer, steps, "") == []
