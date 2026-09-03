"""Minimal Streamlit user interface for document chat."""
from __future__ import annotations

from typing import Any

import streamlit as st

from rag_chat.client import ConfigurationError, get_usage, reset_usage
from rag_chat.ingestion import extract_text, index_text
from rag_chat.query_log import clear_log, log_query, read_log
from rag_chat.retrieval import generate_response, retrieve
from rag_chat.store import get_collection


def format_usage(usage: dict) -> str:
    """WEEK-6 CHANGE: render one question's token/cost usage as a caption line."""
    return (
        f"Tokens: {usage['prompt_tokens']} in + {usage['completion_tokens']} out "
        f"({usage['total_tokens']} total, {usage['calls']} API call(s)) -- "
        f"Est. cost: ${usage['cost_usd']:.5f}"
    )


def initialize_state() -> None:
    """Create session values used to render the chat history and upload list."""
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("documents", [])


def add_upload(uploaded_file: Any) -> None:
    """Index a Streamlit upload and remember its display summary for this session."""
    result = index_text(extract_text(uploaded_file, uploaded_file.name), uploaded_file.name)
    st.session_state.documents.append(result)


def render_chat_tab() -> None:
    """Render the document upload area and the chat itself."""
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
            chunks, sources = [], []
            reset_usage()  # WEEK-6 CHANGE
            try:
                if get_collection().count() == 0:
                    answer = "Upload a document first."
                else:
                    with st.spinner("Searching documents..."):
                        chunks, sources = retrieve(prompt)
                        answer = generate_response(prompt, chunks)
                st.write(answer)
                if sources:
                    st.caption("Sources: " + ", ".join(sources))
            except ConfigurationError as error:
                answer = str(error)
                st.error(answer)
            usage = get_usage()  # WEEK-6 CHANGE
            if usage.calls:
                st.caption(format_usage(usage.__dict__))
        st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})
        # WEEK-6 CHANGE: persist the full trace (not just the chat bubble) so
        # it survives past this browser session and shows up in the Query Log tab.
        log_query(prompt, chunks, sources, answer, usage)

    if st.session_state.messages and st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()


def render_query_log_tab() -> None:
    """WEEK-6 CHANGE: show every question ever asked in this app, most recent first."""
    entries = read_log()
    if not entries:
        st.caption("No queries logged yet -- ask something in the Chat tab.")
        return

    st.caption(f"{len(entries)} quer{'y' if len(entries) == 1 else 'ies'} logged.")

    # WEEK-6 CHANGE: running total across every logged query, so cost/tokens
    # are visible at a glance without opening each entry.
    total_tokens = sum(entry["usage"]["total_tokens"] for entry in entries if entry.get("usage"))
    total_cost = sum(entry["usage"]["cost_usd"] for entry in entries if entry.get("usage"))
    if total_tokens:
        st.caption(f"Total so far: {total_tokens} tokens -- est. ${total_cost:.5f}")

    if st.button("Clear log"):
        clear_log()
        st.rerun()

    for entry in entries:
        with st.expander(f"{entry['timestamp']} -- {entry['question']}"):
            st.write("**Answer:**", entry["answer"])
            if entry["sources"]:
                st.caption("Sources: " + ", ".join(entry["sources"]))
            if entry.get("usage"):  # WEEK-6 CHANGE -- older log lines may not have this
                st.caption(format_usage(entry["usage"]))
            if entry["chunks"]:
                st.caption("Retrieved chunks:")
                for index, chunk in enumerate(entry["chunks"], start=1):
                    st.text(f"[{index}] {chunk}")


def main() -> None:
    """Render the beginner-friendly document upload, chat, and query log interface."""
    st.set_page_config(page_title="Chat with documents")
    initialize_state()
    st.title("Chat with your documents")
    st.caption("1. Upload a TXT, PDF, or DOCX file.  2. Add it to the knowledge base.  3. Ask a question.")

    chat_tab, log_tab = st.tabs(["Chat", "Query Log"])
    with chat_tab:
        render_chat_tab()
    with log_tab:
        render_query_log_tab()


if __name__ == "__main__":
    main()