"""Week 5 error-analysis harness: run a fixed question set through the real
retrieval + generation pipeline and record a complete, replayable trace for
each one (question, fetched chunk text, sources, final answer).

Not part of the app itself -- a throwaway script for producing
docs/week5-traces.json, which docs/week5-error-analysis.md is written from.
"""
from __future__ import annotations

import json
from pathlib import Path


from rag_chat.config import get_settings
from rag_chat.ingestion import index_directory
from rag_chat.retrieval import generate_response, retrieve

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = PROJECT_ROOT / "docs" / "week5-traces.json"

# 13 questions carried over from Week 4 (docs/week4-wrong-query-findings.md)
# plus ~12 new ones covering areas that set didn't touch: cross-currency,
# hours, unique-item lookups, dietary/vegan, signature dishes, and a couple
# more "should refuse" negatives -- so this isn't just a re-run of known
# failures.
QUESTIONS = [
    # --- carried over from Week 4 ---
    "What is the price of the Wagyu burger at Grand Vista Hotel?",
    "What is Sri Kaveri Bhavan's semolina dessert?",
    "What is Vaigai Tiffin Center's famous drink?",
    "Who serves Chettinad Chicken Meals?",
    "What is the price of Filter Coffee at Thanjai Ruchi Mess?",
    "What is the price of Plain Idli at Vaigai Tiffin Center?",
    "What is the price of Medu Vada at Sri Lakshmi Vilas?",
    "What is the price of Ven Pongal at Sri Kaveri Bhavan?",
    "What is the price of Plain Idli at Thanjai Ruchi Mess?",
    "What is the price of Butter Chicken at Thanjai Ruchi Mess?",
    "What is the price of filter coffee in Thanjavur?",
    "Which shop has the cheapest filter coffee, and how much is it?",
    "How much does filter coffee cost at \"Ruchi Mess\"?",
    # --- new for Week 5 ---
    "What time does Sri Lakshmi Vilas open?",
    "What currency does Grand Vista Hotel price its menu in?",
    "What is Kari Dosai and where can I get it?",
    "What is the priciest item on any menu, and which restaurant sells it?",
    "Which shop has the cheapest Medu Vada, and what does each shop charge for it?",
    "What is Thanjai Ruchi Mess's signature tiffin dish?",
    "How much for a Pongal at Lakshmi Vilas?",
    "What vegan items are available across these menus?",
    "What's a good vegetarian breakfast option?",
    "What is the difference in Filter Coffee price between Thanjai Ruchi Mess and Sri Kaveri Bhavan?",
    "Does Sri Kaveri Bhavan serve pizza?",
    "What's on the menu at the restaurant on Gandhi Road?",
]


def main() -> None:
    indexed = index_directory(get_settings().articles_dir)
    if indexed:
        print(f"Checked {len(indexed)} document(s); only new content was embedded.")

    traces = []
    for number, question in enumerate(QUESTIONS, start=1):
        print(f"[{number}/{len(QUESTIONS)}] {question}")
        chunks, sources = retrieve(question)
        answer = generate_response(question, chunks)
        traces.append(
            {
                "number": number,
                "question": question,
                "sources": sources,
                "chunks": chunks,
                "answer": answer,
            }
        )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(traces, indent=2), encoding="utf-8")
    print(f"\nWrote {len(traces)} traces to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
