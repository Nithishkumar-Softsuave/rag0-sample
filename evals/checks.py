"""Rule-based checks -- free, deterministic, no LLM call. Run these first."""
from __future__ import annotations

REFUSAL_PHRASES = (
    "i do not know",
    "i don't know",
    "could not find",
    "cannot find",
    "no such item",
    "does not sell",
    "does not serve",
    "does not have",
    "no information",
    "not available on",
    "not on the menu",
)


def is_refusal(answer: str) -> bool:
    """Whether an answer reads as a refusal / "I don't know"-style response."""
    lowered = answer.lower()
    return any(phrase in lowered for phrase in REFUSAL_PHRASES)


def check_sources(expected_sources: list[str], actual_sources: list[str]) -> bool:
    """All expected source documents must be among the ones actually retrieved."""
    return set(expected_sources).issubset(set(actual_sources))


def check_refusal_behavior(should_refuse: bool, answer: str) -> bool:
    """A should-refuse question must get a refusal; a real question must not."""
    return is_refusal(answer) == should_refuse
