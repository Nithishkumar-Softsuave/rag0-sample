"""OpenRouter client and embedding helpers."""
from functools import lru_cache

from openai import OpenAI

from rag_chat.config import get_settings


class ConfigurationError(RuntimeError):
    """Raised when required application configuration is missing."""


@lru_cache(maxsize=1)
def get_client() -> OpenAI:
    """Create the OpenRouter-compatible OpenAI client only when it is needed."""
    settings = get_settings()
    if not settings.api_key.strip():
        raise ConfigurationError(
            "Missing OPENROUTER_API_KEY. Add it to the .env file next to pyproject.toml."
        )
    return OpenAI(base_url="https://openrouter.ai/api/v1", api_key=settings.api_key)


def get_embeddings(texts: list[str]) -> list[list[float]]:
    """Create embeddings, retrying one input at a time if batching is unsupported."""
    model = get_settings().embedding_model
    client = get_client()
    try:
        response = client.embeddings.create(model=model, input=texts)
        return [item.embedding for item in response.data]
    except Exception:
        return [client.embeddings.create(model=model, input=text).data[0].embedding for text in texts]