"""Tests for pure agent helpers that do not require an API key."""
from unittest.mock import patch

import agents.agent as agent_module
from agents.fixed_workflow import extract_items
from agents.tools import convert_currency, load_catalog, search_menus
from rag_chat.client import Usage


def test_catalog_parses_every_menu_file() -> None:
    """All 5 menu files should contribute at least one record each."""
    shops = {record.shop for record in load_catalog()}
    assert shops == {
        "Grand Vista Hotel", "Sri Kaveri Bhavan", "Sri Lakshmi Vilas",
        "Thanjai Ruchi Mess", "Vaigai Tiffin Center",
    }


def test_search_menus_finds_all_four_filter_coffees() -> None:
    matches = search_menus("Filter Coffee")
    assert {match["shop"] for match in matches} == {
        "Sri Kaveri Bhavan", "Sri Lakshmi Vilas", "Thanjai Ruchi Mess", "Vaigai Tiffin Center",
    }


def test_search_menus_no_match_returns_empty() -> None:
    assert search_menus("Uttapam") == []


def test_convert_currency_same_currency_is_identity() -> None:
    assert convert_currency(100, "INR", "INR") == {"amount": 100, "currency": "INR"}


def test_convert_currency_sgd_to_inr() -> None:
    result = convert_currency(34, "SGD", "INR")
    assert result["currency"] == "INR"
    assert result["amount"] == 2074.0


def test_convert_currency_unsupported_pair_reports_error() -> None:
    result = convert_currency(10, "USD", "INR")
    assert "error" in result


def test_extract_items_matches_exact_item_names() -> None:
    items = extract_items("Which is cheaper: the Grand Vista Wagyu Beef Burger or the Chettinad Chicken Meals?")
    assert "Grand Vista Wagyu Beef Burger" in items
    assert "Chettinad Chicken Meals" in items


def test_extract_items_ignores_catalog_serving_size_notes() -> None:
    """"Medu Vada (2 pcs)" in the catalog should still match a plain "Medu Vada" question."""
    items = extract_items("Which shop has the cheapest Medu Vada?")
    assert "Medu Vada" in items


def test_extract_items_misses_paraphrased_names() -> None:
    """The fixed workflow's known brittleness: a loose paraphrase finds nothing."""
    items = extract_items("Is Sri Lakshmi Vilas's chicken meal pricier than the hotel's beef burger?")
    assert items == []


def test_run_agent_stops_on_cost_budget_without_calling_the_model() -> None:
    """Both stop checks run before the model call each iteration -- confirmed by
    making the model call itself raise, so a pass here proves the loop never
    reached it, not just that the reported stop reason looked right.
    """
    with patch.object(agent_module, "get_usage", return_value=Usage(cost_usd=999.0)), \
         patch.object(agent_module, "create_chat_completion", side_effect=AssertionError("should not be called")):
        result = agent_module.run_agent("irrelevant question", max_cost_usd=0.02)
    assert result.stopped_reason == "max_cost"


def test_run_agent_stops_on_time_budget_without_calling_the_model() -> None:
    with patch.object(agent_module, "create_chat_completion", side_effect=AssertionError("should not be called")):
        result = agent_module.run_agent("irrelevant question", max_seconds=0.0)
    assert result.stopped_reason == "max_seconds"
