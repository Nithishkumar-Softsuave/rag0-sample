"""LLM-as-judge for the one thing a rule can't check well here: whether the
app's answer is factually correct AND attached to the right named entity
(shop/restaurant). A rule can check "is Rs. 20 in the answer" but not "is
Rs. 20 the right shop's Rs. 20" -- that needs judgement, which is exactly
Week 5's Group A/B failure pattern.

Binary verdict (CORRECT/INCORRECT), not 1-10: for this task the question
is "did it get the fact and the entity right", which is a yes/no call, not
a matter of degree. See docs/week6-evals.md for why binary was chosen over
a 1-10 scale.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

from rag_chat.client import create_chat_completion, record_usage, resolve_max_tokens
from rag_chat.config import get_settings

JUDGE_SYSTEM_PROMPT = (
    "You are grading whether an AI assistant's answer to a question is factually "
    "correct, using a reference ground-truth answer written by a human. Pay very "
    "close attention to which named entity (restaurant, shop, or document) each "
    "fact is attributed to: an answer that states a correct-looking number or "
    "fact but attaches it to the wrong shop/restaurant is INCORRECT, not correct, "
    "even if that number appears somewhere in the ground truth. An answer that is "
    "less detailed than the ground truth but does not contradict it is CORRECT. "
    "Respond with only a JSON object: "
    '{"verdict": "CORRECT" or "INCORRECT", "reason": "<one short sentence>"}.'
)


@dataclass(frozen=True)
class JudgeResult:
    verdict: str  # "CORRECT" or "INCORRECT"
    reason: str


def build_judge_prompt(question: str, ground_truth: str, answer: str) -> str:
    return (
        f"Question: {question}\n\n"
        f"Reference ground truth: {ground_truth}\n\n"
        f"Assistant's answer to grade: {answer}"
    )


def parse_verdict(raw_response: str) -> JudgeResult:
    """Extract the verdict JSON, falling back to INCORRECT if unparsable.

    Failing closed (unparsable => INCORRECT, not skipped) means a broken judge
    call shows up as a score drop you'll notice, instead of silently vanishing
    from the results.
    """
    start = raw_response.find("{")
    end = raw_response.rfind("}")
    if start == -1 or end == -1 or end < start:
        return JudgeResult("INCORRECT", "Judge reply was not parsable JSON.")
    try:
        parsed = json.loads(raw_response[start : end + 1])
    except json.JSONDecodeError:
        return JudgeResult("INCORRECT", "Judge reply was not parsable JSON.")
    verdict = str(parsed.get("verdict", "")).strip().upper()
    if verdict not in {"CORRECT", "INCORRECT"}:
        verdict = "INCORRECT"
    return JudgeResult(verdict, str(parsed.get("reason", "")))


def judge_answer(question: str, ground_truth: str, answer: str) -> JudgeResult:
    """Ask the judge model to grade one answer against its ground truth."""
    model = get_settings().chat_model
    response = create_chat_completion(
        model=model,
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": build_judge_prompt(question, ground_truth, answer)},
        ],
        max_tokens=resolve_max_tokens(200),  # a {"verdict": ..., "reason": "<one short sentence>"} object is tiny
    )
    record_usage(response.usage, model)
    return parse_verdict(response.choices[0].message.content or "")
