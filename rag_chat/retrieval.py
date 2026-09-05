"""Hybrid semantic and keyword retrieval plus grounded answer generation."""
from __future__ import annotations

import math
import re
from collections import Counter

from rag_chat.client import get_client, get_embeddings, record_usage
from rag_chat.config import get_settings
from rag_chat.reranking import rerank  # WEEK-4 CHANGE
from rag_chat.store import get_collection


def tokenize(text: str) -> list[str]:
    """Convert text into simple lowercase keyword tokens."""
    return re.findall(r"[a-z0-9]+", text.lower())


def semantic_search(question: str, limit: int) -> list[str]:
    """Return chunk IDs ranked by embedding similarity."""
    collection = get_collection()
    count = collection.count()
    if count == 0:
        return []
    result = collection.query(query_embeddings=[get_embeddings([question])[0]], n_results=min(limit, count))
    return result["ids"][0]


def keyword_search(question: str, limit: int) -> list[str]:
    """Return chunk IDs ranked with a small in-memory BM25 implementation."""
    stored = get_collection().get(include=["documents"])
    documents = stored["documents"]
    if not documents:
        return []
    token_lists = [tokenize(document) for document in documents]
    lengths = [len(tokens) for tokens in token_lists]
    average_length = sum(lengths) / len(lengths)
    document_frequency = Counter(token for tokens in token_lists for token in set(tokens))
    scores: list[float] = []
    for tokens, length in zip(token_lists, lengths):
        counts = Counter(tokens)
        score = 0.0
        for term in tokenize(question):
            frequency = counts[term]
            if not frequency:
                continue
            inverse_frequency = math.log(1 + (len(documents) - document_frequency[term] + 0.5) / (document_frequency[term] + 0.5))
            score += inverse_frequency * (frequency * 2.5) / (frequency + 1.5 * (1 - 0.75 + 0.75 * length / average_length))
        scores.append(score)
    ranked = sorted(range(len(scores)), key=lambda index: scores[index], reverse=True)
    return [stored["ids"][index] for index in ranked[:limit]]


def reciprocal_rank_fusion(rank_lists: list[list[str]], k: int = 60) -> list[str]:
    """Merge ranked lists so semantic and exact-keyword matches both matter."""
    scores: Counter[str] = Counter()
    for ranked_ids in rank_lists:
        for rank, chunk_id in enumerate(ranked_ids, start=1):
            scores[chunk_id] += 1 / (k + rank)
    return [chunk_id for chunk_id, _ in scores.most_common()]


def retrieve(question: str, results: int | None = None, pool: int | None = None) -> tuple[list[str], list[str]]:
    """Return the most relevant chunk text and their source filenames."""
    settings = get_settings()
    results = settings.retrieval_top_k if results is None else results
    pool = settings.retrieval_pool if pool is None else pool
    collection = get_collection()
    if collection.count() == 0:
        return [], []

    # This part is unchanged from week 3: hybrid search decides which
    # documents are even in the running.
    fused_ids = reciprocal_rank_fusion([semantic_search(question, pool), keyword_search(question, pool)])

    # WEEK-4 CHANGE: instead of trusting the fused order directly and slicing
    # straight to `results`, take a wider window of fused candidates and let
    # the reranker (rag_chat/reranking.py) judge each one against the full
    # question before trimming down to `results`. This is what tells two
    # near-duplicate chunks (e.g. two shops' "Filter Coffee" entries) apart.
    candidate_ids = fused_ids[: settings.rerank_candidates]
    candidates = collection.get(ids=candidate_ids, include=["documents", "metadatas"])
    text_by_id = dict(zip(candidates["ids"], candidates["documents"]))
    metadata_by_id = dict(zip(candidates["ids"], candidates["metadatas"]))
    ordered_candidates = [chunk_id for chunk_id in candidate_ids if chunk_id in text_by_id]

    chunk_ids = rerank(
        question,
        ordered_candidates,
        [text_by_id[chunk_id] for chunk_id in ordered_candidates],
        top_n=results,
    )

    sources = sorted({metadata_by_id[chunk_id]["source"] for chunk_id in chunk_ids if chunk_id in metadata_by_id})

    # WEEK-6 CHANGE: label each chunk with the source document it came from.
    # Before, generate_response() saw only raw chunk text with no indication
    # of which document a passage belonged to -- if a chunk didn't happen to
    # repeat the restaurant's name in that section, the model had no way to
    # tell two shops' "Filter Coffee" lines apart. See docs/week6-evals.md
    # (Group A, near-duplicate entity mix-up) and
    # docs/week5-error-analysis.md for the failures this targets.
    labeled_chunks = [
        f"[Source: {metadata_by_id[chunk_id]['source']}]\n{text_by_id[chunk_id]}"
        for chunk_id in chunk_ids
        if chunk_id in text_by_id
    ]
    return labeled_chunks, sources


def generate_response(question: str, chunks: list[str]) -> str:
    """Ask the chat model to answer only from the retrieved document context."""
    if not chunks:
        return "I could not find any relevant information in the indexed documents."
    context = "\n\n".join(chunks)
    model = get_settings().chat_model
    response = get_client().chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "Answer using only the supplied context. Each passage below is labeled "
                    "with its source document as [Source: filename]. If the question names a "
                    "specific entity (a restaurant, shop, or document), use only the passage(s) "
                    "whose [Source: ...] label matches that entity -- ignore facts from other "
                    "sources even if they describe a similar-sounding item. If the question asks "
                    "about multiple entities, or a comparison across all of them, address each "
                    "source separately by name; never blend facts from different sources into one "
                    "unattributed number. Refer to each entity by its real name as it appears "
                    "inside the passage text -- never write the literal '[Source: ...]' tag in "
                    "your answer, that label is only for you to tell passages apart. If the "
                    "context does not contain the answer, say you do not know. Keep the answer "
                    "concise."  # WEEK-6 CHANGE
                ),
            },
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion:\n{question}"},
        ],
        max_tokens=500,  # WEEK-6 CHANGE: answers are asked to stay concise
    )
    record_usage(response.usage, model)  # WEEK-6 CHANGE
    return response.choices[0].message.content or "I could not generate an answer."