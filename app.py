"""Chat with your documents (RAG) — OpenRouter edition.

Standalone terminal app: indexes the .txt files in ./news_articles into
ChromaDB (only new chunks, so re-runs are cheap), then answers your questions
from them with an LLM. No backend, no web UI, no database.

Setup (once):
    python -m venv .venv
    .venv\\Scripts\\Activate.ps1
    pip install -r requirements.txt
    # then paste your key into .env:  OPENROUTER_API_KEY=sk-or-...

Run:
    python app.py                      # interactive Q&A loop
    python app.py "your question"      # one-shot answer

Model choice lives in .env as OpenRouter slugs — change them any time:
    OPENROUTER_CHAT_MODEL   (default: openai/gpt-4o-mini)
    OPENROUTER_EMBED_MODEL  (default: openai/text-embedding-3-small)
"""
import os
import sys

import chromadb
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
CHAT_MODEL = os.getenv("OPENROUTER_CHAT_MODEL", "openai/gpt-4o-mini")
EMBED_MODEL = os.getenv("OPENROUTER_EMBED_MODEL", "openai/text-embedding-3-small")

ARTICLES_DIR = "./news_articles"
STORAGE_DIR = "chroma_persistent_storage"
# New collection name: the old "document_qa_collection" was built with the
# OpenAI-direct embedding function and is left untouched.
COLLECTION_NAME = "docs_openrouter"

_client: OpenAI | None = None


def get_client() -> OpenAI:
    """Lazily create the OpenRouter client so the module imports without a key
    (require_key() guards actual API use)."""
    global _client
    if _client is None:
        _client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=OPENROUTER_API_KEY,
        )
    return _client

chroma_client = chromadb.PersistentClient(path=STORAGE_DIR)
collection = chroma_client.get_or_create_collection(
    name=COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
)


def require_key():
    """Fail fast with a clear message instead of an API 401 stack trace."""
    if not OPENROUTER_API_KEY.strip():
        sys.exit(
            "Missing OPENROUTER_API_KEY. Paste your OpenRouter key into the "
            ".env file next to this script, then run again."
        )


def load_documents_from_directory(directory_path):
    """Read every .txt file in the directory into {id, text} dicts."""
    documents = []
    for filename in os.listdir(directory_path):
        if filename.endswith(".txt"):
            with open(
                os.path.join(directory_path, filename), "r", encoding="utf-8"
            ) as file:
                documents.append({"id": filename, "text": file.read()})
    return documents


def split_text(text, chunk_size=1000, chunk_overlap=200):
    """Split text into overlapping char-window chunks."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - chunk_overlap
    return [chunk for chunk in chunks if chunk.strip()]


def get_embeddings(texts):
    """Embed a list of texts via OpenRouter (batched, per-text fallback)."""
    try:
        response = get_client().embeddings.create(model=EMBED_MODEL, input=texts)
        return [item.embedding for item in response.data]
    except Exception:
        # Some providers reject batched input — fall back to one at a time.
        return [
            get_client().embeddings.create(model=EMBED_MODEL, input=text).data[0].embedding
            for text in texts
        ]


def index_documents():
    """Embed and store any chunks not already in the collection."""
    documents = load_documents_from_directory(ARTICLES_DIR)
    chunked_documents = []
    for doc in documents:
        for i, chunk in enumerate(split_text(doc["text"])):
            chunked_documents.append({"id": f"{doc['id']}_chunk{i+1}", "text": chunk})

    existing_ids = set(collection.get()["ids"])
    new_chunks = [c for c in chunked_documents if c["id"] not in existing_ids]
    if not new_chunks:
        print(f"Index up to date ({len(existing_ids)} chunks).")
        return

    print(f"Embedding {len(new_chunks)} new chunks from {len(documents)} documents...")
    embeddings = get_embeddings([c["text"] for c in new_chunks])
    collection.upsert(
        ids=[c["id"] for c in new_chunks],
        documents=[c["text"] for c in new_chunks],
        embeddings=embeddings,
    )
    print(f"Indexed {collection.count()} total chunks.")


def query_documents(question, n_results=3):
    """Return the most relevant chunks for a question."""
    question_embedding = get_embeddings([question])[0]
    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=min(n_results, collection.count()),
    )
    ids = results["ids"][0]
    chunks = results["documents"][0]
    sources = sorted({chunk_id.rsplit("_chunk", 1)[0] for chunk_id in ids})
    return chunks, sources


def generate_response(question, relevant_chunks):
    """Answer the question with the LLM, grounded in the retrieved chunks."""
    context = "\n\n".join(relevant_chunks)
    response = get_client().chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an assistant for question-answering tasks. Use the "
                    "provided context to answer. If you don't know the answer, say "
                    "that you don't know. Keep the answer concise."
                ),
            },
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion:\n{question}",
            },
        ],
    )
    return response.choices[0].message.content


def ask(question):
    chunks, sources = query_documents(question)
    if not chunks:
        print("No documents indexed yet.")
        return
    answer = generate_response(question, chunks)
    print(f"\nAssistant: {answer}")
    print(f"Sources: {', '.join(sources)}")


def main():
    require_key()
    index_documents()

    if len(sys.argv) > 1:
        ask(" ".join(sys.argv[1:]))
        return

    print('Ask questions about your documents. Type "exit" to quit.')
    while True:
        question = input("\nYou: ").strip()
        if question.lower() in {"exit", "quit"}:
            break
        if question:
            ask(question)


if __name__ == "__main__":
    main()
