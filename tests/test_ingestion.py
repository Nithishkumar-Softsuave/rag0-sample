"""Tests for pure ingestion helpers that do not require an API key."""
from rag_chat.ingestion import split_text


def test_split_text_keeps_overlap() -> None:
    """Adjacent chunks share the configured overlap."""
    chunks = split_text("abcdefghij", chunk_size=6, chunk_overlap=2)
    assert chunks == ["abcdef", "efghij", "ij"]


def test_split_text_rejects_invalid_overlap() -> None:
    """A non-advancing chunk size raises a clear error."""
    try:
        split_text("text", chunk_size=5, chunk_overlap=5)
    except ValueError as error:
        assert "larger" in str(error)
    else:
        raise AssertionError("Expected ValueError")