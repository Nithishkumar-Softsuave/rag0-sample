"""OpenRouter client and embedding helpers."""
from dataclasses import dataclass
from functools import lru_cache

from openai import OpenAI

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
    """Create the OpenRouter-compatible OpenAI client only when it is needed."""
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