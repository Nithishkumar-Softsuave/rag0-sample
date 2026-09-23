"""Trajectory evaluation: did the agent call the tools this task actually
needs, not just did it land on a correct-looking final answer?

A right answer can hide a wrong path -- an extra call that got lucky, a
needed tool skipped in favor of the model's own unverified guess, or a tool
called with bad arguments that happened not to matter this time. This
checks tool *choice* against what the task requires (order-insensitive --
searching item A before B or B before A are equally valid). It cannot catch
a right tool called with wrong arguments; that still needs a human reading
the step log, which is why every step is kept rather than thrown away (see
agents/race.py, which persists them).
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from agents.agent import AgentStep


@dataclass(frozen=True)
class TrajectoryVerdict:
    ok: bool
    actual_tools: list[str]
    expected_tools: list[str]
    note: str


def check_trajectory(steps: list[AgentStep], expected_tools: list[str]) -> TrajectoryVerdict:
    """Compare the tools actually called against the tools the task needs."""
    actual_tools = [step.tool for step in steps if step.tool]
    ok = sorted(actual_tools) == sorted(expected_tools)
    if ok:
        note = "matches expected tool sequence"
    elif len(actual_tools) > len(expected_tools):
        note = "extra/redundant tool call(s) beyond what the task needs"
    elif len(actual_tools) < len(expected_tools):
        note = "skipped a tool call the task needs -- likely answered from an unverified guess"
    else:
        note = "used the wrong tool(s) for what the task needs"
    return TrajectoryVerdict(ok=ok, actual_tools=actual_tools, expected_tools=expected_tools, note=note)


def tool_choice_scores(steps: list[AgentStep], expected_tools: list[str]) -> tuple[float, float]:
    """Tool-choice (precision, recall) against the tools the task needs.

    Precision: of the calls the agent made, how many were needed (padding and
    wrong tools lower it). Recall: of the calls the task needs, how many the
    agent made (skipped tools lower it). Multiset-based, so a needed second
    search_menus call counts separately from the first.
    """
    actual = Counter(step.tool for step in steps if step.tool)
    expected = Counter(expected_tools)
    matched = sum(min(actual[tool], count) for tool, count in expected.items())
    precision = matched / sum(actual.values()) if actual else (1.0 if not expected else 0.0)
    recall = matched / sum(expected.values()) if expected else 1.0
    return precision, recall
