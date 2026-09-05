"""Week 6 test set.

These are the same 25 questions used for Week 5's error analysis
(`scripts/collect_traces.py`, `docs/week5-error-analysis.md`) -- not a new
sample. Turning them into a structured, re-runnable test set is the point of
this week: the 8 that failed become permanent regression tests (tagged with
the named failure group they belong to), and the 17 that passed stay in the
set as guardrails so a future change can't quietly break them.

`human_verdict` records the human grading already done in Week 5 (the
Verdict column of the results table). It is reused as-is, not re-graded, to
validate the LLM judge in `evals/validate_judge.py`.
"""
from __future__ import annotations

from dataclasses import dataclass, field

HOTEL = "hotel_menu_rag_sample.txt"
KAVERI = "sri_kaveri_bhavan_menu.txt"
LAKSHMI = "sri_lakshmi_vilas_menu.txt"
THANJAI = "thanjai_ruchi_mess_menu.txt"
VAIGAI = "vaigai_tiffin_center_menu.txt"


@dataclass(frozen=True)
class TestCase:
    """One eval case: a question, how to grade it, and its Week 5 history."""

    id: int
    question: str
    expected_sources: list[str]
    ground_truth: str
    human_verdict: str  # "correct" or "incorrect" -- from docs/week5-error-analysis.md
    should_refuse: bool = False
    failure_group: str | None = None  # A-E, or None if it passed in Week 5
    notes: str = ""


TEST_CASES: list[TestCase] = [
    TestCase(
        1, "What is the price of the Wagyu burger at Grand Vista Hotel?",
        [HOTEL], "The Grand Vista Wagyu Beef Burger is $34.00.", "correct",
    ),
    TestCase(
        2, "What is Sri Kaveri Bhavan's semolina dessert?",
        [KAVERI], "Rava Kesari, priced at Rs. 35.", "correct",
    ),
    TestCase(
        3, "What is Vaigai Tiffin Center's famous drink?",
        [VAIGAI], "Madurai Jigarthanda, priced at Rs. 70.", "correct",
    ),
    TestCase(
        4, "Who serves Chettinad Chicken Meals?",
        [LAKSHMI], "Sri Lakshmi Vilas, priced at Rs. 220.", "correct",
    ),
    TestCase(
        5, "What is the price of Filter Coffee at Thanjai Ruchi Mess?",
        [THANJAI], "Rs. 20.", "incorrect", failure_group="A",
        notes="Week 5: answered Rs. 25, which is Sri Kaveri Bhavan's price.",
    ),
    TestCase(
        6, "What is the price of Plain Idli at Vaigai Tiffin Center?",
        [VAIGAI], "Rs. 30.", "correct",
    ),
    TestCase(
        7, "What is the price of Medu Vada at Sri Lakshmi Vilas?",
        [LAKSHMI], "Rs. 42.", "correct",
    ),
    TestCase(
        8, "What is the price of Ven Pongal at Sri Kaveri Bhavan?",
        [KAVERI], "Rs. 55.", "correct",
    ),
    TestCase(
        9, "What is the price of Plain Idli at Thanjai Ruchi Mess?",
        [THANJAI], "Rs. 35.", "correct",
    ),
    TestCase(
        10, "What is the price of Butter Chicken at Thanjai Ruchi Mess?",
        [THANJAI], "Thanjai Ruchi Mess does not sell Butter Chicken -- the menu has no such item.",
        "correct", should_refuse=True,
    ),
    TestCase(
        11, "What is the price of filter coffee in Thanjavur?",
        [KAVERI, THANJAI],
        "Both Thanjavur shops sell it: Sri Kaveri Bhavan at Rs. 25, Thanjai Ruchi Mess at Rs. 20.",
        "incorrect", failure_group="A",
        notes="Week 5: answered Rs. 22, which is Vaigai Tiffin Center's price -- and Vaigai is in Madurai, not Thanjavur.",
    ),
    TestCase(
        12, "Which shop has the cheapest filter coffee, and how much is it?",
        [KAVERI, LAKSHMI, THANJAI, VAIGAI],
        "Thanjai Ruchi Mess has the cheapest Filter Coffee at Rs. 20 (vs. Rs. 22 Vaigai, Rs. 24 Sri Lakshmi Vilas, Rs. 25 Sri Kaveri Bhavan).",
        "incorrect", failure_group="A",
        notes="Week 5: landed on the right price (Rs. 20) but never named which shop it belongs to.",
    ),
    TestCase(
        13, 'How much does filter coffee cost at "Ruchi Mess"?',
        [THANJAI], "Rs. 20 (Thanjai Ruchi Mess).", "incorrect", failure_group="A",
        notes="Week 5: answered Rs. 25, which is Sri Kaveri Bhavan's price, despite the right document being retrieved.",
    ),
    TestCase(
        14, "What time does Sri Lakshmi Vilas open?",
        [LAKSHMI], "6:30 AM.", "correct",
    ),
    TestCase(
        15, "What currency does Grand Vista Hotel price its menu in?",
        [HOTEL], "Singapore Dollar (SGD).", "correct",
    ),
    TestCase(
        16, "What is Kari Dosai and where can I get it?",
        [VAIGAI],
        "A dosa stuffed with spiced Madurai-style mutton kari filling, Rs. 90, a Vaigai Tiffin Center specialty.",
        "correct",
    ),
    TestCase(
        17, "What is the priciest item on any menu, and which restaurant sells it?",
        [HOTEL],
        "Pan-Seared Atlantic Salmon, $38.00, at Grand Vista Hotel (correct once SGD vs. INR is accounted for).",
        "correct",
        notes="Graded as a pass in Week 5 but flagged as a near-miss: cross-currency comparison isn't done explicitly.",
    ),
    TestCase(
        18, "Which shop has the cheapest Medu Vada, and what does each shop charge for it?",
        [KAVERI, LAKSHMI, THANJAI, VAIGAI],
        "Vaigai Tiffin Center is cheapest at Rs. 35. Full list: Sri Kaveri Bhavan Rs. 45, Sri Lakshmi Vilas Rs. 42, Thanjai Ruchi Mess Rs. 40, Vaigai Tiffin Center Rs. 35.",
        "incorrect", failure_group="C",
        notes="Week 5: correct cheapest shop, but the per-shop list silently dropped Sri Kaveri Bhavan (RETRIEVAL_TOP_K=3 truncation) -- out of scope for this week's fix.",
    ),
    TestCase(
        19, "What is Thanjai Ruchi Mess's signature tiffin dish?",
        [THANJAI], "Thanjavur Vengaya Dosai.", "correct",
    ),
    TestCase(
        20, "How much for a Pongal at Lakshmi Vilas?",
        [LAKSHMI], "Rs. 58.", "correct",
    ),
    TestCase(
        21, "What vegan items are available across these menus?",
        [KAVERI, VAIGAI, HOTEL],
        "Buttermilk (Sri Kaveri Bhavan, vegan option available), Paruthi Paal (Vaigai Tiffin Center, vegan), "
        "and Selection of TWG Teas (Grand Vista Hotel, vegan). Idiyappam is NOT tagged vegan in any source.",
        "incorrect", failure_group="E",
        notes="Week 5: invented a 'vegan' tag for Idiyappam (source only says Vegetarian, Gluten-Free) and missed TWG Teas entirely.",
    ),
    TestCase(
        22, "What's a good vegetarian breakfast option?",
        [HOTEL],
        "Any vegetarian breakfast item is acceptable if grounded in a real menu entry, e.g. Avocado & Poached Egg Toast at Grand Vista Hotel.",
        "correct",
    ),
    TestCase(
        23, "What is the difference in Filter Coffee price between Thanjai Ruchi Mess and Sri Kaveri Bhavan?",
        [KAVERI, THANJAI], "Rs. 5 (Rs. 25 minus Rs. 20).", "incorrect", failure_group="D",
        notes="Week 5: said 'I do not know' -- both docs were retrieved, but the specific chunk with Thanjai's Rs. 20 line wasn't in context.",
    ),
    TestCase(
        24, "Does Sri Kaveri Bhavan serve pizza?",
        [KAVERI], "No -- Sri Kaveri Bhavan's menu has no pizza.", "correct", should_refuse=True,
    ),
    TestCase(
        25, "What's on the menu at the restaurant on Gandhi Road?",
        [KAVERI],
        "Only Sri Kaveri Bhavan's own items: Plain Idli, Masala Dosa, Medu Vada, Ven Pongal, Rava Kesari, "
        "Kaveri Special Thali, Curd Rice, Filter Coffee, Buttermilk.",
        "incorrect", failure_group="B",
        notes="Week 5: blended in Vaigai Tiffin Center's Kari Dosai/Jigarthanda and Sri Lakshmi Vilas's "
        "Chettinad Chicken/Badam Milk under Sri Kaveri Bhavan's name -- the worst failure in the batch.",
    ),
]

FAILURE_GROUP_NAMES = {
    "A": "Near-duplicate entity mix-up",
    "B": "Cross-document menu bleeding",
    "C": "Top-k truncation drops sources",
    "D": "Chunk boundary hides the answer",
    "E": "Fabricated attributes",
}
