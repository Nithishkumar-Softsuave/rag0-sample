"""Terminal entry point for document chat."""
from __future__ import annotations

import argparse

from rag_chat.client import ConfigurationError
from rag_chat.config import get_settings
from rag_chat.ingestion import index_directory
from rag_chat.retrieval import generate_response, retrieve


def ask(question: str) -> None:
    """Print an answer and its sources for one question."""
    chunks, sources = retrieve(question)
    print(f"\nAssistant: {generate_response(question, chunks)}")
    if sources:
        print(f"Sources: {', '.join(sources)}")


def main() -> None:
    """Index bundled documents, then run one question or an interactive chat."""
    parser = argparse.ArgumentParser(description="Ask questions about local documents.")
    parser.add_argument("question", nargs="*", help="Optional one-shot question.")
    args = parser.parse_args()
    try:
        indexed = index_directory(get_settings().articles_dir)
        if indexed:
            print(f"Checked {len(indexed)} document(s); only new content was embedded.")
        if args.question:
            ask(" ".join(args.question))
            return
        print("Ask questions about your documents. Type 'exit' to quit.")
        while True:
            question = input("\nYou: ").strip()
            if question.lower() in {"exit", "quit"}:
                return
            if question:
                ask(question)
    except ConfigurationError as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()