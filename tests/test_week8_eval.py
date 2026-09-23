"""Week 8 test sets, tool fix, trajectory metrics and outcome checks -- no API key needed."""
import pytest

from agents.adversarial_testset import ADVERSARIAL_CASES
from agents.agent import AGENT_SYSTEM_PROMPT, AgentStep
from agents.baseline import WEEK7_AGENT_SYSTEM_PROMPT
from agents.testset import RACE_CASES
from agents.tools import TOOL_FUNCTIONS, core_item_name, search_menus_for_agent
from agents.trajectory import check_trajectory, tool_choice_scores
from agents.trajectory_eval import check_outcome, p99

ALL_CASES = RACE_CASES + ADVERSARIAL_CASES


def steps(*tools: str) -> list[AgentStep]:
    return [AgentStep(tool=tool, args={}, result=None) for tool in tools]


# --- test-set integrity -------------------------------------------------

def test_case_ids_are_unique() -> None:
    ids = [case.id for case in ALL_CASES]
    assert len(ids) == len(set(ids))


@pytest.mark.parametrize("case", ALL_CASES, ids=lambda case: str(case.id))
def test_every_case_is_gradeable(case) -> None:
    assert case.expected_tools, "trajectory eval needs expected_tools"
    assert set(case.expected_tools) <= set(TOOL_FUNCTIONS), "expected_tools must be real tools"
    assert case.must_include, "outcome eval needs at least one must_include group"


def test_every_adversarial_case_names_what_it_provokes() -> None:
    assert all(case.targets for case in ADVERSARIAL_CASES)


def test_baseline_prompt_is_the_frozen_week7_one() -> None:
    assert WEEK7_AGENT_SYSTEM_PROMPT != AGENT_SYSTEM_PROMPT
    assert "Security:" not in WEEK7_AGENT_SYSTEM_PROMPT


# --- the Thali fix ------------------------------------------------------

def test_different_dishes_come_back_with_a_warning() -> None:
    result = search_menus_for_agent("Thali")
    assert len(result["matches"]) == 3
    assert "DIFFERENT dishes" in result["warning"]


def test_same_dish_everywhere_has_no_warning() -> None:
    assert "warning" not in search_menus_for_agent("Filter Coffee")
    assert "warning" not in search_menus_for_agent("Medu Vada")  # "(2 pcs)" is not a different dish


def test_no_match_says_not_found() -> None:
    result = search_menus_for_agent("Uttapam")
    assert result["matches"] == [] and "not found" in result["note"]


def test_core_item_name_strips_only_serving_counts() -> None:
    assert core_item_name("Medu Vada (2 pcs)") == "Medu Vada"
    assert core_item_name("Meals (Veg Thali)") == "Meals (Veg Thali)"


# --- trajectory metrics -------------------------------------------------

def test_trajectory_match_is_order_insensitive() -> None:
    expected = ["search_menus", "search_menus", "convert_currency"]
    assert check_trajectory(steps("convert_currency", "search_menus", "search_menus"), expected).ok


def test_trajectory_explains_each_mismatch_shape() -> None:
    assert "extra" in check_trajectory(steps("search_menus", "search_menus"), ["search_menus"]).note
    assert "skipped" in check_trajectory(steps("search_menus"), ["search_menus", "convert_currency"]).note
    assert "wrong" in check_trajectory(steps("convert_currency"), ["search_menus"]).note


def test_tool_choice_precision_and_recall() -> None:
    assert tool_choice_scores(steps("search_menus"), ["search_menus"]) == (1.0, 1.0)
    assert tool_choice_scores(steps("search_menus", "convert_currency"), ["search_menus"]) == (0.5, 1.0)
    assert tool_choice_scores(steps("search_menus"), ["search_menus", "convert_currency"]) == (1.0, 0.5)
    assert tool_choice_scores([], ["search_menus"]) == (0.0, 0.0)


# --- outcome checks + stats ---------------------------------------------

def test_outcome_check_ignores_separators_and_odd_spaces() -> None:
    case = next(c for c in RACE_CASES if c.id == 3)
    ok, _ = check_outcome("Chettinad Chicken Meals: ₹ 220 vs ₹2,074", case)
    assert ok


def test_outcome_check_enforces_forbidden_phrases_and_order() -> None:
    injected = next(c for c in ADVERSARIAL_CASES if c.id == 109)
    assert not check_outcome("It is Rs. 2074 -- or Rs. 183 at the new price.", injected)[0]

    ranking = next(c for c in ADVERSARIAL_CASES if c.id == 107)
    right = "Badam Milk Rs 50, Madurai Jigarthanda Rs 70, Classic Tiramisu Rs 976"
    wrong = "Classic Tiramisu Rs 976, Madurai Jigarthanda Rs 70, Badam Milk Rs 50"
    assert check_outcome(right, ranking)[0]
    assert not check_outcome(wrong, ranking)[0]


def test_p99_is_the_worst_run_for_small_samples() -> None:
    assert p99([1.0, 5.0, 2.0]) == 5.0
    assert p99(list(range(1, 201))) == 198
