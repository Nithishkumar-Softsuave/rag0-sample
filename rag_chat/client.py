"""OpenRouter client and embedding helpers."""
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from functools import lru_cache

from openai import OpenAI, RateLimitError

from rag_chat.config import get_settings


class ConfigurationError(RuntimeError):
    """Raised when required application configuration is missing."""


# WEEK-6 CHANGE: local token/cost tracking, so a single question's full cost
# (embedding + rerank + generation calls combined) can be shown in the app
# without needing a LangSmith account. Prices are OpenAI's published list
# prices per 1M tokens -- an estimate, since OpenRouter's actual billed rate
# can differ slightly by provider/model routing.
PRICING_PER_MILLION_TOKENS = {
    "openai/gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "openai/text-embedding-3-small": {"input": 0.02, "output": 0.0},
    # WEEK-10 CHANGE: the active chat model (LLM_PROVIDER=groq) was missing,
    # so every Week 7-9 run reported $0.00 -- the cost column of a race was
    # meaningless. Groq's list price (Oct 2026); the free tier bills $0, so
    # this is what the same traffic would cost on a paid plan.
    "openai/gpt-oss-120b": {"input": 0.15, "output": 0.60},
}


@dataclass
class Usage:
    """Accumulated token usage and estimated cost for one request.

    WEEK-10 CHANGE: usage scopes nest. A call recorded inside a scope also
    counts toward every enclosing scope (`parent`), so a multi-agent run's
    total includes every specialist's calls -- see `usage_scope()`.
    """

    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float = 0.0
    # WEEK-10 CHANGE: time spent sleeping on Groq rate limits (see
    # create_chat_completion). It inflates wall-clock latency without saying
    # anything about the agent, so a speed comparison has to see it.
    rate_limit_waits: int = 0
    rate_limit_wait_seconds: float = 0.0
    parent: "Usage | None" = field(default=None, repr=False, compare=False)


# WEEK-10 CHANGE: was a module global that every run_agent() reset at its
# start -- a specialist agent started by a manager agent wiped the manager's
# running total, so a team's tokens and cost came out far too low. A
# ContextVar gives each thread/scope its own current total.
_current_usage: ContextVar[Usage] = ContextVar("current_usage", default=Usage())
# Specialists run in parallel threads and add to the same parent totals.
_usage_lock = threading.Lock()


def reset_usage() -> None:
    """Start a fresh, top-level usage total -- call before handling one question."""
    _current_usage.set(Usage())


def get_usage() -> Usage:
    """Return the usage accumulated in the current scope (since the last reset)."""
    return _current_usage.get()


@contextmanager
def usage_scope() -> Iterator[Usage]:
    """Track one unit of work's usage without losing the enclosing total.

    Calls recorded inside are counted in the yielded Usage *and* in every
    enclosing scope, so nested agents roll up into their caller's total.
    Threads start with an empty context: run worker code with
    `contextvars.copy_context().run(...)` so it nests under the caller.
    """
    usage = Usage(parent=_current_usage.get())
    token = _current_usage.set(usage)
    try:
        yield usage
    finally:
        _current_usage.reset(token)


def record_usage(response_usage, model: str) -> None:
    """Add one API call's token usage (and estimated cost) to the current scope and its parents."""
    if response_usage is None:
        return
    prompt_tokens = getattr(response_usage, "prompt_tokens", 0) or 0
    completion_tokens = getattr(response_usage, "completion_tokens", 0) or 0
    total_tokens = getattr(response_usage, "total_tokens", 0) or (prompt_tokens + completion_tokens)
    rates = PRICING_PER_MILLION_TOKENS.get(model, {"input": 0.0, "output": 0.0})
    cost = (prompt_tokens / 1_000_000) * rates["input"] + (completion_tokens / 1_000_000) * rates["output"]

    with _usage_lock:
        usage: Usage | None = _current_usage.get()
        while usage is not None:
            usage.calls += 1
            usage.prompt_tokens += prompt_tokens
            usage.completion_tokens += completion_tokens
            usage.total_tokens += total_tokens
            usage.cost_usd += cost
            usage = usage.parent


def _record_rate_limit_wait(seconds: float) -> None:
    """Count one rate-limit backoff sleep in the current scope and its parents."""
    with _usage_lock:
        usage: Usage | None = _current_usage.get()
        while usage is not None:
            usage.rate_limit_waits += 1
            usage.rate_limit_wait_seconds += seconds
            usage = usage.parent


@lru_cache(maxsize=1)
def get_client() -> OpenAI:
    """The OpenRouter client -- always used for embeddings (get_embeddings()
    below), and for chat completions too unless an alternate provider is
    configured (see get_chat_client()).
    """
    settings = get_settings()
    if not settings.api_key.strip():
        raise ConfigurationError(
            "Missing OPENROUTER_API_KEY. Add it to the .env file next to pyproject.toml."
        )
    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=settings.api_key)

    # WEEK-6 CHANGE: wrap the client so every chat/embeddings call is traced to
    # LangSmith. Safe with no LangSmith account at all -- wrap_openai only
    # emits traces when LANGSMITH_TRACING=true and LANGSMITH_API_KEY are set
    # in .env; otherwise it's a harmless passthrough.
    if settings.langsmith_tracing:
        from langsmith.wrappers import wrap_openai

        client = wrap_openai(client)
    return client


def resolve_max_tokens(base: int) -> int:
    """TEMP: every chat-completion call site should wrap its `max_tokens`
    literal in this instead of passing it bare.

    `base` was tuned tight (200-500) for OpenRouter's gpt-4o-mini, on
    purpose: a low-balance OpenRouter account gets a 402 if `max_tokens`
    asks for more than the account can afford, even if the model wouldn't
    have used that many. Groq's gpt-oss-120b is a reasoning model, though --
    it spends tokens on hidden reasoning before writing anything visible,
    and `base` alone leaves zero room for the actual answer (confirmed:
    500 came back completely empty; 2000 was comfortable). Groq's free
    tier has no billing reason to keep this tight, so scale up generously
    whenever it's the active provider.
    """
    return 2000 if get_settings().llm_provider == "groq" else base


@lru_cache(maxsize=1)
def get_chat_client() -> OpenAI:
    """TEMP: the client every *chat completions* call site should use --
    reranking.py, retrieval.py's generate_response(), evals/judge.py,
    agents/agent.py. Defaults to get_client() (OpenRouter), same as before
    this existed. Set LLM_PROVIDER=groq in .env to route chat completions to
    Groq instead (free tier, useful for verifying the Week 7 agent while
    OpenRouter credits are out) -- get_embeddings() below is deliberately
    untouched by this setting, since Groq has no embeddings endpoint.
    """
    settings = get_settings()
    if settings.llm_provider != "groq":
        return get_client()
    if not settings.groq_api_key.strip():
        raise ConfigurationError(
            "Missing GROQ_API_KEY. Add it to .env, or remove LLM_PROVIDER (or set it to "
            "'openrouter') to use OpenRouter for chat instead."
        )
    return OpenAI(base_url="https://api.groq.com/openai/v1", api_key=settings.groq_api_key)


def create_chat_completion(**kwargs):
    """TEMP: what every chat-completion call site should call instead of
    `get_chat_client().chat.completions.create(...)` directly.

    Groq's free tier has a tight tokens-per-minute cap that a handful of
    back-to-back agent/judge calls (each now needing a much bigger
    max_tokens for gpt-oss-120b's hidden reasoning, see resolve_max_tokens
    above) can hit well before anything is actually wrong. The cap resets
    every minute, so a short backoff and retry is enough -- this is not
    needed (and barely triggers) against OpenRouter, whose calls stay small.
    """
    client = get_chat_client()
    delay = 5.0
    for attempt in range(4):
        try:
            return client.chat.completions.create(**kwargs)
        except RateLimitError:
            if attempt == 3:
                raise
            _record_rate_limit_wait(delay)
            time.sleep(delay)
            delay *= 2


def get_embeddings(texts: list[str]) -> list[list[float]]:
    """Create embeddings, retrying one input at a time if batching is unsupported."""
    model = get_settings().embedding_model
    client = get_client()
    try:
        response = client.embeddings.create(model=model, input=texts)
        record_usage(response.usage, model)  # WEEK-6 CHANGE
        return [item.embedding for item in response.data]
    except Exception:
        embeddings = []
        for text in texts:
            response = client.embeddings.create(model=model, input=text)
            record_usage(response.usage, model)  # WEEK-6 CHANGE
            embeddings.append(response.data[0].embedding)
        return embeddings