"""6 questions for the same task -- "compare a menu item's price, converting
currency where needed" -- chosen to expose exactly where a fixed sequence
keeps up with an agent, and where it can't.

Week 8 adds two things to every case: `expected_tools` (what a correct
*path* looks like, for trajectory evaluation) and deterministic outcome
checks (`must_include` / `must_not_include` / `ordered`), so a batch of
dozens of runs can be graded without paying for -- or trusting -- an LLM
judge on every one. agents/race.py still uses the judge + `ground_truth`,
exactly as in Week 7. Week 8's adversarial cases live in
agents/adversarial_testset.py.
"""
from __future__ import annotations

from dataclasses import dataclass

# Phrases that mean "I looked, and it isn't there" -- shared by every case
# whose correct answer is a not-found.
# Widened after reading the first full run by hand ("I wasn't able to find a
# menu entry for..." is a correct not-found and was graded a failure).
# Curly apostrophes are normalized away by trajectory_eval._normalize.
NOT_FOUND = (
    "not on", "not found", "isn't on", "not listed", "isn't listed", "couldn't find", "could not find",
    "wasn't able to find", "wasn't able to locate", "unable to find", "unable to locate",
    "doesn't", "does not", "no listing", "not available", "no such", "no match", "no matching",
    "don't have", "do not have", "not sold", "isn't sold", "not offered",
)


@dataclass(frozen=True)
class RaceCase:
    id: int
    question: str
    ground_truth: str
    kind: str  # simple / cross_currency / paraphrase / not_found / multi_fact / Week 8 adversarial kinds
    notes: str = ""
    # The tool calls this question actually requires, order-insensitive --
    # e.g. two items to look up plus one conversion. Used by
    # agents/trajectory.py to catch a right answer reached via a wrong (or
    # skipped, or padded) path, not just to grade the final text.
    expected_tools: tuple[str, ...] = ()
    # Deterministic outcome check (agents/trajectory_eval.py: check_outcome).
    # must_include: every group must match; a group matches if ANY of its
    # phrases appears (case-insensitive, commas and odd spaces ignored).
    must_include: tuple[tuple[str, ...], ...] = ()
    must_not_include: tuple[str, ...] = ()
    ordered: tuple[str, ...] = ()  # these must first appear in this order
    targets: str = ""  # which failure mode an adversarial case is built to provoke


RACE_CASES: list[RaceCase] = [
    RaceCase(
        1, "Which shop has the cheapest Filter Coffee?",
        "Thanjai Ruchi Mess, at Rs. 20 -- the cheapest of the four shops that sell it (vs. Rs. 22, 24, 25).",
        "simple", notes="Exact item name, single currency (INR). Both paths should handle this equally well.",
        expected_tools=("search_menus",),
        must_include=(("thanjai",), ("20",)),
    ),
    RaceCase(
        2, "Which shop has the cheapest Medu Vada?",
        "Vaigai Tiffin Center, at Rs. 35 -- the cheapest of the four shops that sell it (vs. Rs. 40, 42, 45).",
        "simple", notes="Same shape as #1, different item -- a second easy case.",
        expected_tools=("search_menus",),
        must_include=(("vaigai",), ("35",)),
    ),
    RaceCase(
        3, "Which is cheaper in Indian Rupees: the Grand Vista Wagyu Beef Burger or the Chettinad Chicken Meals?",
        "The Chettinad Chicken Meals, sold by Sri Lakshmi Vilas (Rs. 220), is cheaper -- the Wagyu Beef "
        "Burger, sold by Grand Vista Hotel, is $34 SGD, about Rs. 2074 at the fixed rate used here, far "
        "more expensive.",
        "cross_currency", notes="Needs the currency conversion tool. Both items are named exactly as they "
        "appear on their menus, so the fixed workflow's substring match should still find them.",
        expected_tools=("search_menus", "search_menus", "convert_currency"),
        must_include=(("chettinad",), ("220",), ("2074",)),
    ),
    RaceCase(
        4, "Is Sri Lakshmi Vilas's chicken meal pricier than the hotel's beef burger, once converted to rupees?",
        "No -- the chicken meal (Rs. 220) is far cheaper than the beef burger "
        "(~Rs. 2074 once converted from $34 SGD).",
        "paraphrase", notes="Same fact as #3, worded loosely ('chicken meal', 'the hotel's beef burger') "
        "instead of the menus' exact item names. Expected to break the fixed workflow's substring "
        "matcher -- this is the case that argues for the agent.",
        expected_tools=("search_menus", "search_menus", "convert_currency"),
        must_include=(("220",), ("2074",), ("cheaper", "not pricier", "less expensive", "more expensive")),
    ),
    RaceCase(
        5, "What's the price of Uttapam at Vaigai Tiffin Center?",
        "Uttapam is not on Vaigai Tiffin Center's menu, or any menu in this corpus -- there is no such item.",
        "not_found", notes="Reliability check: a nonexistent item should be reported as not found, not guessed at.",
        expected_tools=("search_menus",),
        must_include=(NOT_FOUND,),
    ),
    RaceCase(
        6, "Compare Filter Coffee prices across every shop and tell me the cheapest and priciest.",
        "Cheapest: Thanjai Ruchi Mess at Rs. 20. Priciest: Sri Kaveri Bhavan at Rs. 25.",
        "multi_fact", notes="One search, but two facts (min and max) instead of one -- checks whether "
        "either path drops the second fact.",
        expected_tools=("search_menus",),
        must_include=(("thanjai",), ("20",), ("kaveri",), ("25",)),
    ),
]
