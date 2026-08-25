"""Minimal Streamlit user interface for document chat."""
from __future__ import annotations

from typing import Any

import streamlit as st

from rag_chat.client import ConfigurationError
from rag_chat.ingestion import extract_text, index_text
from rag_chat.retrieval import generate_response, retrieve
from rag_chat.store import get_collection


def initialize_state() -> None:
    """Create session values used to render the chat history and upload list."""
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("documents", [])


def add_upload(uploaded_file: Any) -> None:
    """Index a Streamlit upload and remember its display summary for this session."""
    result = index_text(extract_text(uploaded_file, uploaded_file.name), uploaded_file.name)
    st.session_state.documents.append(result)


def main() -> None:
    """Render the beginner-friendly document upload and chat interface."""
    st.set_page_config(page_title="Chat with documents")
    initialize_state()
    st.title("Chat with your documents")
    st.caption("1. Upload a TXT, PDF, or DOCX file.  2. Add it to the knowledge base.  3. Ask a question.")

    uploads = st.file_uploader("Upload documents", type=["txt", "pdf", "docx"], accept_multiple_files=True)
    if uploads and st.button("Add to knowledge base", type="primary"):
        for upload in uploads:
            try:
                with st.spinner(f"Indexing {upload.name}..."):
                    add_upload(upload)
                st.success(f"Added {upload.name}.")
            except (ConfigurationError, ValueError) as error:
                st.error(str(error))
            except Exception:
                st.error(f"Could not index {upload.name}. Check that it is a readable document and your OpenRouter settings are valid.")

    if st.session_state.documents:
        names = ", ".join(document.source for document in st.session_state.documents)
        st.caption(f"This session: {names}")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
            if message.get("sources"):
                st.caption("Sources: " + ", ".join(message["sources"]))

    if prompt := st.chat_input("Ask a question about your documents"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)
        with st.chat_message("assistant"):
            try:
                if get_collection().count() == 0:
                    answer, sources = "Upload a document first.", []
                else:
                    with st.spinner("Searching documents..."):
                        chunks, sources = retrieve(prompt)
                        answer = generate_response(prompt, chunks)
                st.write(answer)
                if sources:
                    st.caption("Sources: " + ", ".join(sources))
            except ConfigurationError as error:
                answer, sources = str(error), []
                st.error(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})

    if st.session_state.messages and st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()


if __name__ == "__main__":
    main()