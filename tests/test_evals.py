"""Tests for pure eval helpers that do not require an API key."""
from evals.checks import check_refusal_behavior, check_sources, is_refusal
from evals.judge import parse_verdict


def test_check_sources_requires_all_expected() -> None:
    """Missing even one expected source fails the check."""
    assert check_sources(["a.txt", "b.txt"], ["a.txt", "b.txt", "c.txt"]) is True
    assert check_sources(["a.txt", "b.txt"], ["a.txt"]) is False


def test_refusal_detection() -> None:
    """Common refusal phrasing is recognized; a real answer is not."""
    assert is_refusal("I do not know.") is True
    assert is_refusal("The price is Rs. 20.") is False


def test_check_refusal_behavior_matches_expectation() -> None:
    assert check_refusal_behavior(should_refuse=True, answer="I do not know.") is True
    assert check_refusal_behavior(should_refuse=True, answer="Rs. 20.") is False
    assert check_refusal_behavior(should_refuse=False, answer="Rs. 20.") is True


def test_parse_verdict_reads_json() -> None:
    result = parse_verdict('Sure: {"verdict": "correct", "reason": "matches ground truth"}')
    assert result.verdict == "CORRECT"
    assert result.reason == "matches ground truth"


def test_parse_verdict_fails_closed_on_malformed_reply() -> None:
    """An unparsable judge reply counts as INCORRECT, not skipped."""
    result = parse_verdict("I refuse to answer in JSON.")
    assert result.verdict == "INCORRECT"
