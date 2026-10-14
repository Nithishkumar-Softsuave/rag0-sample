"""Week 11: one trace per request, one span per step -- so any past answer can be found and explained.

Before this, the only record of a request was rag_chat/query_log.py's line
for the RAG chat tab: question, chunks, answer and one lump of usage. The
agent and the squad left nothing behind once the terminal scrolled away. A
customer saying "it gave me a wrong answer yesterday" could not be followed
up at all.

Now every request through agents/kitchen.py writes one JSON line to
data/traces.jsonl:

    request_id, timestamp, route, user/session, question, answer, outcome,
    model/provider, app version, prompt version, MCP tool fingerprints,
    total time, tokens, cost -- and a list of spans:

    request                      9.1s  $0.00061
      agent                      9.0s  $0.00061
        llm                      3.2s  $0.00030   (tokens, finish_reason)
        tool:lookup_ingredients  0.1s             (args, result, blocked?)
        llm                      5.6s  $0.00031
        validate                 0.0s             (issues)

Spans nest through a ContextVar, the same way usage scopes do (Week 10), so
a squad's specialists, running in worker threads under
contextvars.copy_context(), land under the right parent. Each span opens its
own usage_scope(), so its tokens and cost are exactly its own subtree.

Outside a request, span() does nothing -- the tests, the race harnesses and
the CLIs that don't log pay nothing for it.

    python -m observability.logs find "dairy"       # see observability/logs.py
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import threading
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any

from rag_chat.client import Usage, usage_scope
from rag_chat.config import PROJECT_ROOT, get_settings

TRACE_PATH = Path(os.getenv("TRACE_LOG_PATH", PROJECT_ROOT / "data" / "traces.jsonl"))

# A tool result or answer is logged in full up to this size: enough to
# replay and audit every result this app's tools return, without one
# runaway document blowing up the log.
MAX_FIELD_CHARS = 4000

_write_lock = threading.Lock()


def new_request_id() -> str:
    return "req_" + uuid.uuid4().hex[:12]


def prompt_version(text: str) -> str:
    """A short hash of a prompt: which wording produced an answer, without logging the prompt."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:10]


@lru_cache(maxsize=1)
def app_version() -> str:
    """The git commit the code ran from (plus '+dirty' for uncommitted changes)."""
    try:
        sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=PROJECT_ROOT,
                             capture_output=True, text=True, timeout=5).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=PROJECT_ROOT,
                               capture_output=True, text=True, timeout=5).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return (sha or "unknown") + ("+dirty" if dirty else "")


def clip(value: Any) -> Any:
    """JSON-safe copy of `value`, with long strings cut to MAX_FIELD_CHARS."""
    if isinstance(value, str):
        return value if len(value) <= MAX_FIELD_CHARS else value[:MAX_FIELD_CHARS] + f"...[+{len(value) - MAX_FIELD_CHARS} chars]"
    if isinstance(value, dict):
        return {str(k): clip(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clip(v) for v in value]
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if hasattr(value, "model_dump"):
        return clip(value.model_dump())
    return clip(str(value))


@dataclass
class Span:
    """One step of a request: an LLM call, a tool call, an agent, a check."""

    id: str
    parent_id: str | None
    name: str
    kind: str                     # "request" | "agent" | "llm" | "tool" | "stage" | "check"
    start_ms: float               # offset from the start of the request
    duration_ms: float = 0.0
    llm_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_prompt_tokens: int = 0
    cost_usd: float = 0.0
    rate_limit_wait_seconds: float = 0.0
    attrs: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    def set(self, **attrs: Any) -> None:
        self.attrs.update(clip(attrs))


@dataclass
class Trace:
    """Everything logged about one request; written as one JSON line."""

    request_id: str
    timestamp: str
    route: str
    question: str
    user_id: str | None = None
    session_id: str | None = None
    answer: str | None = None
    outcome: str | None = None          # the run's stopped_reason, or "error"
    duration_ms: float = 0.0
    llm_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_prompt_tokens: int = 0
    cost_usd: float = 0.0
    rate_limit_wait_seconds: float = 0.0
    attrs: dict[str, Any] = field(default_factory=dict)
    spans: list[Span] = field(default_factory=list)
    _start: float = field(default=0.0, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def set(self, **attrs: Any) -> None:
        self.attrs.update(clip(attrs))

    def to_json(self) -> dict:
        # Not asdict(self): it deep-copies every field, and a Lock can't be copied.
        data = {f.name: getattr(self, f.name) for f in fields(self) if not f.name.startswith("_")}
        data["spans"] = [asdict(s) for s in self.spans]
        return data


_current_trace: ContextVar[Trace | None] = ContextVar("current_trace", default=None)
_current_span: ContextVar[Span | None] = ContextVar("current_span", default=None)


def current_trace() -> Trace | None:
    return _current_trace.get()


def _copy_usage(target: Span | Trace, usage: Usage) -> None:
    target.llm_calls = usage.calls
    target.prompt_tokens = usage.prompt_tokens
    target.completion_tokens = usage.completion_tokens
    target.cached_prompt_tokens = usage.cached_prompt_tokens
    target.cost_usd = round(usage.cost_usd, 8)
    target.rate_limit_wait_seconds = usage.rate_limit_wait_seconds
    if usage.malformed_tool_call_retries:
        target.attrs["malformed_tool_call_retries"] = usage.malformed_tool_call_retries


@contextmanager
def request(question: str, *, route: str, user_id: str | None = None, session_id: str | None = None,
            path: Path | None = None, **attrs: Any) -> Iterator[Trace]:
    """Trace one request end to end and append it to the log, even if it raised."""
    settings = get_settings()
    trace = Trace(
        request_id=new_request_id(), timestamp=datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
        route=route, question=question, user_id=user_id, session_id=session_id, _start=time.perf_counter(),
    )
    trace.set(model=settings.chat_model, provider=settings.llm_provider, app_version=app_version(), **attrs)
    trace_token = _current_trace.set(trace)
    try:
        with span("request", kind="request", route=route) as root, usage_scope() as usage:
            try:
                yield trace
            except BaseException as exc:
                trace.outcome = "error"
                trace.set(error=f"{type(exc).__name__}: {exc}")
                raise
            finally:
                if root is not None:
                    root.set(outcome=trace.outcome)
    finally:
        _current_trace.reset(trace_token)
        trace.duration_ms = round((time.perf_counter() - trace._start) * 1000, 1)
        _copy_usage(trace, usage)
        write(trace, path or TRACE_PATH)


@contextmanager
def span(name: str, kind: str = "stage", **attrs: Any) -> Iterator[Span | None]:
    """Time one step and count its tokens/cost; yields None (and records nothing) outside a request."""
    trace = _current_trace.get()
    if trace is None:
        yield None
        return
    parent = _current_span.get()
    item = Span(id=uuid.uuid4().hex[:8], parent_id=parent.id if parent else None, name=name, kind=kind,
                start_ms=round((time.perf_counter() - trace._start) * 1000, 1))
    item.set(**attrs)
    with trace._lock:  # added at the start, so the log lists steps in the order they began
        trace.spans.append(item)
    span_token = _current_span.set(item)
    started = time.perf_counter()
    try:
        with usage_scope() as usage:
            yield item
    except BaseException as exc:
        item.error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        _current_span.reset(span_token)
        item.duration_ms = round((time.perf_counter() - started) * 1000, 1)
        _copy_usage(item, usage)


def write(trace: Trace, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(trace.to_json(), ensure_ascii=False)
    with _write_lock, path.open("a", encoding="utf-8") as file:
        file.write(line + "\n")


def read_traces(path: Path | None = None) -> list[dict]:
    """Every logged request, oldest first. A torn last line (crash mid-write) is skipped, not fatal."""
    path = path or TRACE_PATH
    if not path.exists():
        return []
    traces = []
    with path.open(encoding="utf-8") as file:
        for line in file:
            if line.strip():
                try:
                    traces.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return traces
