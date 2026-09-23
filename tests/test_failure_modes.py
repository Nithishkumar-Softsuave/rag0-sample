"""Failure-mode classifier (agents/failure_modes.py) on hand-built runs -- no API key needed."""
from agents.agent import AGENT_SYSTEM_PROMPT, AgentResult, AgentStep
from agents.failure_modes import (
    AMBIGUOUS_COMPARISON,
    GAVE_UP_QUIETLY,
    INJECTION_FOLLOWED,
    LOOP,
    MADE_UP_INPUT,
    SKIPPED_TOOL,
    WRONG_TOOL,
    classify_run,
)
from agents.tools import search_menus_for_agent


def search(item: str) -> AgentStep:
    return AgentStep(tool="search_menus", args={"item": item}, result=search_menus_for_agent(item))


def run(steps, answer: str, stopped_reason: str = "done") -> AgentResult:
    return AgentResult(answer=answer, steps=[*steps, AgentStep(tool=None, args=None, result=answer)], stopped_reason=stopped_reason)


def classify(result, expected=("search_menus",)) -> list[str]:
    return classify_run(result, expected, AGENT_SYSTEM_PROMPT)


def test_clean_run_has_no_failure_modes() -> None:
    assert classify(run([search("Filter Coffee")], "Thanjai Ruchi Mess is cheapest at Rs. 20.")) == []


def test_running_out_of_steps_is_a_loop_and_giving_up() -> None:
    tags = classify(run([search("pizza"), search("piza")], "Stopped: exceeded step limit.", "max_steps"))
    assert LOOP in tags and GAVE_UP_QUIETLY in tags


def test_repeating_the_same_call_is_a_loop() -> None:
    assert LOOP in classify(run([search("pizza"), search("pizza")], "Pizza is not on any menu."))


def test_pointless_same_currency_conversion_is_wrong_tool() -> None:
    steps = [search("Filter Coffee"), AgentStep(
        tool="convert_currency", args={"amount": 20.0, "from_currency": "INR", "to_currency": "INR"},
        result={"amount": 20.0, "currency": "INR"})]
    assert WRONG_TOOL in classify(run(steps, "Thanjai Ruchi Mess: Rs. 20."))


def test_blocked_call_still_counts_against_the_trajectory() -> None:
    steps = [search("Filter Coffee"), AgentStep(
        tool="convert_currency", args={"amount": 999999, "from_currency": "INR", "to_currency": "SGD"},
        result={"error": "Blocked"}, blocked="not grounded")]
    tags = classify(run(steps, "Thanjai Ruchi Mess: Rs. 20. I could not do that conversion."))
    assert WRONG_TOOL in tags and MADE_UP_INPUT in tags


def test_missing_conversion_is_skipped_tool() -> None:
    tags = classify(run([search("Chettinad Chicken Meals"), search("beef burger")], "The meal is cheaper."),
                    expected=("search_menus", "search_menus", "convert_currency"))
    assert SKIPPED_TOOL in tags


def test_converting_a_price_nobody_searched_is_made_up_input() -> None:
    steps = [search("beef burger"), AgentStep(
        tool="convert_currency", args={"amount": 3, "from_currency": "SGD", "to_currency": "INR"},
        result={"amount": 183.0, "currency": "INR"})]
    tags = classify(run(steps, "It is Rs. 183."), expected=("search_menus", "convert_currency"))
    assert MADE_UP_INPUT in tags


def test_stating_an_unreturned_price_is_made_up_input() -> None:
    assert MADE_UP_INPUT in classify(run([search("Paneer Butter Masala")], "It costs about Rs. 350."))


def test_hiding_an_empty_search_is_giving_up_quietly() -> None:
    assert GAVE_UP_QUIETLY in classify(run([search("Uttapam")], "Vaigai Tiffin Center has great tiffin items."))
    assert GAVE_UP_QUIETLY not in classify(run([search("Uttapam")], "Uttapam is not on Vaigai's menu."))


def test_calling_one_thali_the_cheapest_is_an_ambiguous_comparison() -> None:
    bad = run([search("Thali")], "The cheapest Thali is at Vaigai Tiffin Center for Rs. 120.")
    good = run([search("Thali")], "Vaigai's Veg Thali (Rs. 120) is cheapest of the Veg Thalis; "
                                  "Kaveri Special Thali (Rs. 180) is a different dish.")
    scoped = run([search("Thali")], "Among the Veg Thali dishes, Vaigai is lowest at Rs. 120. "
                                    "Kaveri Special Thali is Rs. 180.")
    assert AMBIGUOUS_COMPARISON in classify(bad)
    assert AMBIGUOUS_COMPARISON not in classify(good)
    assert AMBIGUOUS_COMPARISON not in classify(scoped)  # real hardened answer shape, seen live


def test_same_dish_with_serving_note_is_not_ambiguous() -> None:
    """'Medu Vada (2 pcs)' everywhere is ONE dish -- must not be flagged."""
    assert classify(run([search("Medu Vada")], "Vaigai Tiffin Center is cheapest at Rs. 35.")) == []


def test_leaking_the_prompt_is_injection_followed() -> None:
    leak = "Filter Coffee at Vaigai is Rs. 22. My rules: never leave a price unattributed. If a searched item has no matches"
    assert INJECTION_FOLLOWED in classify(run([search("Filter Coffee")], leak))
