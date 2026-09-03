"""WEEK-6 CHANGE: append-only log of every question asked, for the in-app log view.

Separate from LangSmith tracing -- this is a lightweight local file so the
Streamlit UI can show a "Query Log" tab without needing a LangSmith account.
"""
from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timezone

from rag_chat.client import Usage
from rag_chat.config import get_settings

LOG_PATH = get_settings().chroma_dir.parent / "query-log.jsonl"


def log_query(question: str, chunks: list[str], sources: list[str], answer: str, usage: Usage | None = None) -> None:
    """Append one query's full trace (plus its token usage/cost, if given) as a JSON line."""
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "question": question,
        "sources": sources,
        "chunks": chunks,
        "answer": answer,
        "usage": asdict(usage) if usage is not None else None,  # WEEK-6 CHANGE
    }
    with LOG_PATH.open("a", encoding="utf-8") as file:
        file.write(json.dumps(entry) + "\n")


def read_log() -> list[dict]:
    """Return every logged query, most recent first."""
    if not LOG_PATH.exists():
        return []
    lines = LOG_PATH.read_text(encoding="utf-8").splitlines()
    entries = [json.loads(line) for line in lines if line.strip()]
    return list(reversed(entries))


def clear_log() -> None:
    """Delete all logged queries."""
    if LOG_PATH.exists():
        LOG_PATH.unlink()
