"""OpenRouter client and embedding helpers."""
import time
from dataclasses import dataclass
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
}


@dataclass
class Usage:
    """Accumulated token usage and estimated cost for one request."""

    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float = 0.0


_usage = Usage()


def reset_usage() -> None:
    """Start a fresh usage total -- call before handling one question."""
    global _usage
    _usage = Usage()


def get_usage() -> Usage:
    """Return the usage accumulated since the last `reset_usage()` call."""
    return _usage


def record_usage(response_usage, model: str) -> None:
    """Add one API call's token usage (and estimated cost) to the running total."""
    if response_usage is None:
        return
    prompt_tokens = getattr(response_usage, "prompt_tokens", 0) or 0
    completion_tokens = getattr(response_usage, "completion_tokens", 0) or 0
    total_tokens = getattr(response_usage, "total_tokens", 0) or (prompt_tokens + completion_tokens)
    rates = PRICING_PER_MILLION_TOKENS.get(model, {"input": 0.0, "output": 0.0})
    cost = (prompt_tokens / 1_000_000) * rates["input"] + (completion_tokens / 1_000_000) * rates["output"]

    _usage.calls += 1
    _usage.prompt_tokens += prompt_tokens
    _usage.completion_tokens += completion_tokens
    _usage.total_tokens += total_tokens
    _usage.cost_usd += cost


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