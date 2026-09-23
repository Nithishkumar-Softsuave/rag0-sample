"""The same task as one hardcoded sequence -- no model decides the steps.

Item names are matched by plain substring against the known catalog, not
by an LLM reading the question. That is the honest trade-off this file is
built to expose: free and instant when the question's wording matches a
menu item closely enough, and unable to answer at all when it doesn't
(see docs/week7-agents.md for where this actually breaks).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

from agents.tools import convert_currency, core_item_name, load_catalog, search_menus


@dataclass
class FixedResult:
    answer: str
    elapsed_seconds: float = 0.0
    cost_usd: float = 0.0  # always 0.0 -- this path never calls the model
    matched_items: list[str] = field(default_factory=list)


def _known_items() -> list[str]:
    """Every distinct item core name in the catalog, longest first.

    Longest-first matters: "Filter Coffee" must be checked before a
    (hypothetical) shorter item name that happens to be its substring,
    so the more specific match wins.
    """
    return sorted({core_item_name(record.item) for record in load_catalog()}, key=len, reverse=True)


def extract_items(question: str) -> list[str]:
    """Which known menu items are named, verbatim, inside the question."""
    lowered = question.lower()
    return [item for item in _known_items() if item.lower() in lowered]


def run_fixed_workflow(question: str) -> FixedResult:
    """Step 1: find named items. Step 2: look up + convert. Step 3: template the answer."""
    start = time.perf_counter()
    items = extract_items(question)
    if not items:
        return FixedResult(
            answer="Could not identify a known menu item in the question wording.",
            elapsed_seconds=time.perf_counter() - start,
        )

    rows: list[tuple[str, str, float, str, float]] = []  # shop, item, price, currency, price_in_inr
    for item in items:
        for record in search_menus(item):
            converted = convert_currency(record["price"], record["currency"], "INR")
            if "error" in converted:
                continue
            rows.append((record["shop"], record["item"], record["price"], record["currency"], converted["amount"]))

    if not rows:
        return FixedResult(answer="No matching menu item found.", elapsed_seconds=time.perf_counter() - start, matched_items=items)

    lines = [f"- {shop}: {item} -- {price:.2f} {currency} (~Rs. {inr:.2f})" for shop, item, price, currency, inr in rows]
    cheapest = min(rows, key=lambda row: row[4])
    answer = "\n".join(lines) + f"\n\nCheapest: {cheapest[0]}'s {cheapest[1]} at {cheapest[2]:.2f} {cheapest[3]}."
    return FixedResult(answer=answer, elapsed_seconds=time.perf_counter() - start, matched_items=items)
