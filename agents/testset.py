"""6 questions for the same task -- "compare a menu item's price, converting
currency where needed" -- chosen to expose exactly where a fixed sequence
keeps up with an agent, and where it can't.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RaceCase:
    id: int
    question: str
    ground_truth: str
    kind: str  # simple / cross_currency / paraphrase / not_found / multi_fact
    notes: str = ""


RACE_CASES: list[RaceCase] = [
    RaceCase(
        1, "Which shop has the cheapest Filter Coffee?",
        "Thanjai Ruchi Mess, at Rs. 20 -- the cheapest of the four shops that sell it (vs. Rs. 22, 24, 25).",
        "simple", notes="Exact item name, single currency (INR). Both paths should handle this equally well.",
    ),
    RaceCase(
        2, "Which shop has the cheapest Medu Vada?",
        "Vaigai Tiffin Center, at Rs. 35 -- the cheapest of the four shops that sell it (vs. Rs. 40, 42, 45).",
        "simple", notes="Same shape as #1, different item -- a second easy case.",
    ),
    RaceCase(
        3, "Which is cheaper in Indian Rupees: the Grand Vista Wagyu Beef Burger or the Chettinad Chicken Meals?",
        "The Chettinad Chicken Meals, sold by Sri Lakshmi Vilas (Rs. 220), is cheaper -- the Wagyu Beef "
        "Burger, sold by Grand Vista Hotel, is $34 SGD, about Rs. 2074 at the fixed rate used here, far "
        "more expensive.",
        "cross_currency", notes="Needs the currency conversion tool. Both items are named exactly as they "
        "appear on their menus, so the fixed workflow's substring match should still find them.",
    ),
    RaceCase(
        4, "Is Sri Lakshmi Vilas's chicken meal pricier than the hotel's beef burger, once converted to rupees?",
        "No -- the chicken meal (Rs. 220) is far cheaper than the beef burger "
        "(~Rs. 2074 once converted from $34 SGD).",
        "paraphrase", notes="Same fact as #3, worded loosely ('chicken meal', 'the hotel's beef burger') "
        "instead of the menus' exact item names. Expected to break the fixed workflow's substring "
        "matcher -- this is the case that argues for the agent.",
    ),
    RaceCase(
        5, "What's the price of Uttapam at Vaigai Tiffin Center?",
        "Uttapam is not on Vaigai Tiffin Center's menu, or any menu in this corpus -- there is no such item.",
        "not_found", notes="Reliability check: a nonexistent item should be reported as not found, not guessed at.",
    ),
    RaceCase(
        6, "Compare Filter Coffee prices across every shop and tell me the cheapest and priciest.",
        "Cheapest: Thanjai Ruchi Mess at Rs. 20. Priciest: Sri Kaveri Bhavan at Rs. 25.",
        "multi_fact", notes="One search, but two facts (min and max) instead of one -- checks whether "
        "either path drops the second fact.",
    ),
]
