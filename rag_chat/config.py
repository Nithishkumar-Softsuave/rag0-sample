"""Application settings loaded from the project-local .env file."""
from dataclasses import dataclass
from pathlib import Path
import os

from dotenv import load_dotenv

# ``config.py`` is in ``<project>/rag_chat``; its parent is the project root.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")


@dataclass(frozen=True)
class Settings:
    """Configuration used by the RAG services."""

    api_key: str
    chat_model: str
    embedding_model: str
    articles_dir: Path
    chroma_dir: Path
    collection_name: str
    chunk_size: int
    chunk_overlap: int
    retrieval_top_k: int
    retrieval_pool: int
    rerank_candidates: int  # WEEK-4 CHANGE
    langsmith_tracing: bool  # WEEK-6 CHANGE
    # TEMP: alternate chat provider for verifying the Week 7 agent while
    # OpenRouter credits are out. Groq has no embeddings endpoint, so this
    # only ever swaps the *chat* client -- retrieve()'s embeddings calls
    # still go to OpenRouter regardless. See rag_chat/client.py:get_client().
    llm_provider: str
    groq_api_key: str
    groq_chat_model: str


def get_settings() -> Settings:
    """Return application settings without exposing the API key in logs."""
    llm_provider = os.getenv("LLM_PROVIDER", "openrouter").strip().lower()
    groq_chat_model = os.getenv("GROQ_CHAT_MODEL", "openai/gpt-oss-120b")
    # `chat_model` resolves to whichever provider is active, so every call
    # site that already reads get_settings().chat_model (reranking.py,
    # retrieval.py, evals/judge.py, agents/agent.py) needs no other change --
    # only get_client() vs get_chat_client() (rag_chat/client.py) differs.
    chat_model = groq_chat_model if llm_provider == "groq" else os.getenv("OPENROUTER_CHAT_MODEL", "openai/gpt-4o-mini")

    return Settings(
        api_key=os.getenv("OPENROUTER_API_KEY", ""),
        chat_model=chat_model,
        embedding_model=os.getenv("OPENROUTER_EMBED_MODEL", "openai/text-embedding-3-small"),
        articles_dir=PROJECT_ROOT / "news_articles",
        chroma_dir=PROJECT_ROOT / "data" / "chroma",
        collection_name="docs_openrouter",
        chunk_size=int(os.getenv("CHUNK_SIZE", "1000")),
        chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "200")),
        retrieval_top_k=int(os.getenv("RETRIEVAL_TOP_K", "3")),
        retrieval_pool=int(os.getenv("RETRIEVAL_POOL", "15")),
        rerank_candidates=int(os.getenv("RETRIEVAL_RERANK_CANDIDATES", "8")),  # WEEK-4 CHANGE
        langsmith_tracing=os.getenv("LANGSMITH_TRACING", "false").lower() == "true",  # WEEK-6 CHANGE
        llm_provider=llm_provider,
        groq_api_key=os.getenv("GROQ_API_KEY", ""),
        groq_chat_model=groq_chat_model,
    )
