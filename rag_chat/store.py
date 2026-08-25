"""ChromaDB persistence for document chunks."""
from functools import lru_cache

import chromadb

from rag_chat.config import get_settings


@lru_cache(maxsize=1)
def get_collection():
    """Return the persistent collection used by both the CLI and Streamlit UI."""
    settings = get_settings()
    settings.chroma_dir.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(settings.chroma_dir))
    return client.get_or_create_collection(
        name=settings.collection_name,
        metadata={"hnsw:space": "cosine"},
    )