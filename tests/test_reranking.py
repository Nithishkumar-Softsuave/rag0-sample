"""WEEK-4 CHANGE: tests for the pure reranking helpers that do not require an API key."""
from rag_chat.reranking import parse_rank_order


def test_parse_rank_order_reads_json_array() -> None:
    """A well-formed model reply is parsed straight into a rank order."""
    assert parse_rank_order("Sure, here you go: [3, 1, 2]", candidate_count=3) == [3, 1, 2]


def test_parse_rank_order_falls_back_on_malformed_reply() -> None:
    """An unparsable reply keeps the original (fused-search) order instead of breaking."""
    assert parse_rank_order("I cannot rank these.", candidate_count=3) == [1, 2, 3]
