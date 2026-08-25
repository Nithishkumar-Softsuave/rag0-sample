# RAG Document Chat

A small, beginner-friendly RAG application. It can index TXT, PDF, and DOCX documents into ChromaDB, retrieve information with hybrid search, and answer through OpenRouter.

## Folder structure

```text
rag0-sample/
├── src/rag_chat/          # Application code
│   ├── config.py          # Environment settings and paths
│   ├── client.py          # OpenRouter client and embeddings
│   ├── ingestion.py       # File reading, chunking, and indexing
│   ├── store.py           # ChromaDB collection
│   ├── retrieval.py       # Hybrid search and answer generation
│   ├── cli.py             # Terminal application
│   └── ui.py              # Streamlit interface
├── news_articles/         # Example documents
├── data/chroma/           # Created automatically; local vector data
├── app.py                 # Terminal entry point
└── streamlit_app.py       # Streamlit entry point
```

## Setup

1. Install Python 3.11 or newer.
2. Create and activate a virtual environment:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
```

3. Install the application:

```powershell
pip install -e .
```

4. Copy `.env.example` to `.env`, then add an OpenRouter key. Never commit `.env`.

```env
OPENROUTER_API_KEY=your_key_here
```

## Run

Start the browser UI:

```powershell
streamlit run streamlit_app.py
```

Use the terminal chat instead:

```powershell
python app.py
python app.py "What is on the menu?"
```

## How it works

1. A file is converted to text.
2. The text is split into overlapping chunks.
3. Chunks are embedded and saved in `data/chroma`.
4. A question uses semantic search and keyword search together.
5. The chat model answers only from the retrieved chunks.

Re-uploading unchanged content does not embed it again. Delete `data/chroma` only if you intentionally want to rebuild the local knowledge base.