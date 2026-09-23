"""Check the agent's final answer before anyone sees it.

Tool guards (agents/guards.py) control what the agent can *do*; this
controls what it can *say*. Three rules, all plain code, no second LLM call:

1. Every money figure in the answer must trace back to a number a tool
   actually returned in this run (or a sum/difference of two of them, so
   "Rs. 1,854 more expensive" is allowed). A figure that appears nowhere in
   the tool results is either made up or was taken from the question
   itself -- the "made-up inputs" failure mode, caught at the output.
2. No links, e-mail/UPI-style payment ids, or canary markers. A menu-price
   agent has no reason to hand a customer a payment link; every one seen in
   testing came from an injected instruction.
3. No verbatim chunk (8+ consecutive words) of the system prompt -- the
   system-prompt-leak attack.

Known limit: rule 1 cannot catch a poisoned document that states a false
price, because that price *is* in the tool results. See
docs/week8-agent-failures.md, "What could still get through".
"""
from __future__ import annotations

import re
from itertools import combinations

from agents.guards import grounded_amounts, is_grounded, price_like_numbers

_LINK = re.compile(r"https?://|www\.", re.IGNORECASE)
_HANDLE = re.compile(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)*\b")  # e-mail or UPI id (name@bank)
_CANARY = re.compile(r"CANARY_", re.IGNORECASE)
_WORD = re.compile(r"[a-z0-9']+")
LEAK_SHINGLE_WORDS = 8


def _allowed_figures(steps) -> set[float]:
    grounded = grounded_amounts(steps)
    allowed = set(grounded)
    for first, second in combinations(sorted(grounded), 2):
        allowed.add(round(second - first, 2))
        allowed.add(round(first + second, 2))
    allowed.update({round(value) for value in list(allowed)})
    return allowed


def ungrounded_figures(answer: str, steps) -> list[float]:
    """Money figures in `answer` that no tool in this run returned (or summed/differenced to)."""
    allowed = _allowed_figures(steps)
    return sorted(value for value in price_like_numbers(answer) if not is_grounded(value, allowed))


def contains_link_or_payment_id(answer: str) -> bool:
    return bool(_LINK.search(answer) or _HANDLE.search(answer))


def contains_canary(answer: str) -> bool:
    return bool(_CANARY.search(answer))


def leaks_system_prompt(answer: str, system_prompt: str) -> bool:
    """True if any 8 consecutive words of the system prompt appear in the answer."""
    prompt_words = _WORD.findall(system_prompt.lower())
    answer_text = " ".join(_WORD.findall(answer.lower()))
    for start in range(len(prompt_words) - LEAK_SHINGLE_WORDS + 1):
        if " ".join(prompt_words[start : start + LEAK_SHINGLE_WORDS]) in answer_text:
            return True
    return False


def validate_answer(answer: str, steps, system_prompt: str) -> list[str]:
    """Return every problem found in `answer`; an empty list means it passed."""
    issues: list[str] = []
    ungrounded = ungrounded_figures(answer, steps)
    if ungrounded:
        figures = ", ".join(f"{value:g}" for value in ungrounded)
        issues.append(f"states money figure(s) {figures} that no tool returned in this run")
    if contains_link_or_payment_id(answer):
        issues.append("contains a link or payment/e-mail id")
    if contains_canary(answer):
        issues.append("contains an injected canary marker")
    if system_prompt and leaks_system_prompt(answer, system_prompt):
        issues.append("repeats part of the system prompt verbatim")
    return issues
