"""WEEK-4 CHANGE: LLM-based reranking -- a second relevance pass after retrieval.

Why this file exists: `retrieval.py`'s hybrid search (semantic + BM25 + RRF)
decides *which documents* come back, but it can't tell two near-duplicate
chunks apart -- e.g. four different menus all selling "Filter Coffee" at
different prices. RRF fusion happily ranks a wrong-shop chunk right next to
the correct one, and the final answer picks whichever one the model happens
to read first.

Reranking fixes that by judging each already-retrieved candidate directly
against the *full* question (including the exact entity name), instead of
comparing independent embeddings. See docs/week4-wrong-query-findings.md,
failures #1 and #4, for the two real queries this was built to fix.
"""
from __future__ import annotations

import json

from rag_chat.client import get_client, record_usage
from rag_chat.config import get_settings

# WEEK-4 CHANGE: the reranker's instructions. Deliberately calls out exact
# names/places, since that is precisely what embedding similarity blurs.
RERANK_SYSTEM_PROMPT = (
    "You are a search relevance judge. You will be given a question and a "
    "numbered list of candidate text passages. Rank the passages from most "
    "to least relevant to answering the question. Pay close attention to "
    "exact names, places, and identifiers in the question -- a passage "
    "about the wrong named entity is not relevant, even if it discusses "
    "the same general topic. Respond with only a JSON array of the "
    "passage numbers, most relevant first, e.g. [3, 1, 2]."
)


def build_rerank_prompt(question: str, chunks: list[str]) -> str:
    """WEEK-4 CHANGE: format the question and numbered candidates for the judge model."""
    numbered = "\n\n".join(f"[{index}] {chunk}" for index, chunk in enumerate(chunks, start=1))
    return f"Question: {question}\n\nCandidate passages:\n{numbered}"


def parse_rank_order(raw_response: str, candidate_count: int) -> list[int]:
    """WEEK-4 CHANGE: extract a 1-based rank order from the model's reply.

    Falls back to the original (unreranked) order if the model's reply
    can't be parsed, so a malformed response degrades gracefully instead
    of breaking retrieval.
    """
    start = raw_response.find("[")
    end = raw_response.rfind("]")
    if start == -1 or end == -1 or end < start:
        return list(range(1, candidate_count + 1))
    try:
        order = json.loads(raw_response[start : end + 1])
    except json.JSONDecodeError:
        return list(range(1, candidate_count + 1))
    return [int(item) for item in order if isinstance(item, (int, float))]


def rerank(question: str, chunk_ids: list[str], chunk_texts: list[str], top_n: int) -> list[str]:
    """WEEK-4 CHANGE: reorder candidate chunk IDs by relevance to the question.

    `chunk_ids` and `chunk_texts` must be the same length and in matching
    order (chunk_texts[i] is the text for chunk_ids[i]). Returns at most
    `top_n` chunk IDs, most relevant first.
    """
    if len(chunk_ids) <= 1:
        return chunk_ids[:top_n]

    model = get_settings().chat_model
    response = get_client().chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": RERANK_SYSTEM_PROMPT},
            {"role": "user", "content": build_rerank_prompt(question, chunk_texts)},
        ],
    )
    record_usage(response.usage, model)  # WEEK-6 CHANGE
    order = parse_rank_order(response.choices[0].message.content or "", len(chunk_ids))
    ranked_ids = [chunk_ids[index - 1] for index in order if 1 <= index <= len(chunk_ids)]

    # Anything the model left out (or an unparsable reply) keeps its original
    # fused-search position instead of being dropped silently.
    remaining = [chunk_id for chunk_id in chunk_ids if chunk_id not in ranked_ids]
    return (ranked_ids + remaining)[:top_n]
