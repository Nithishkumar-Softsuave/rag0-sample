"""One-command race: run the agent and the fixed workflow on the same 6
questions, grade both with Week 6's judge, and compare speed, cost, and
correctness.

The judge itself turned out to be non-deterministic run to run (the same
deterministic fixed-workflow answer got graded CORRECT once and INCORRECT
the next run, with zero code change in between) -- the same lesson Week 6
already learned about single-pass evals. `--repeat` re-judges (and
re-runs the agent) N times per case and scores by majority vote, so that
noise is visible instead of hidden behind one lucky/unlucky pass.

Usage:
    python -m agents.race                # one pass per case
    python -m agents.race --repeat 3      # 3 passes per case, majority vote
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

from agents.agent import run_agent
from agents.fixed_workflow import run_fixed_workflow
from agents.testset import RACE_CASES, RaceCase
from agents.trajectory import check_trajectory
from evals.judge import judge_answer

RESULTS_PATH = Path(__file__).resolve().parent / "results" / "race.json"


@dataclass
class RaceRow:
    id: int
    question: str
    kind: str
    agent_answer: str
    agent_correct: str  # CORRECT / INCORRECT -- majority vote across trials
    agent_pass_rate: float
    agent_seconds: float
    agent_cost_usd: float
    agent_tool_calls: int
    agent_stopped_reason: str
    fixed_answer: str
    fixed_correct: str  # majority vote too -- the judge, not the answer, is what's flaky
    fixed_pass_rate: float
    fixed_seconds: float
    fixed_cost_usd: float
    trials: int
    # Trajectory eval (Week 8): did every trial call the tools this question
    # actually needs, not just land on a correct-looking answer?
    trajectory_ok: bool = True
    trajectory_notes: list[str] = field(default_factory=list)
    agent_steps: list[list[dict]] = field(default_factory=list)  # one step list per trial


def run_case(case: RaceCase, repeat: int = 1) -> RaceRow:
    agent_passes: list[bool] = []
    fixed_passes: list[bool] = []
    trajectory_oks: list[bool] = []
    trajectory_notes: list[str] = []
    agent_steps: list[list[dict]] = []
    agent_result = fixed_result = None

    for _ in range(repeat):
        agent_result = run_agent(case.question)
        agent_judged = judge_answer(case.question, case.ground_truth, agent_result.answer)
        agent_passes.append(agent_judged.verdict == "CORRECT")

        verdict = check_trajectory(agent_result.steps, list(case.expected_tools))
        trajectory_oks.append(verdict.ok)
        trajectory_notes.append(verdict.note)
        agent_steps.append([asdict(step) for step in agent_result.steps])

        fixed_result = run_fixed_workflow(case.question)
        fixed_judged = judge_answer(case.question, case.ground_truth, fixed_result.answer)
        fixed_passes.append(fixed_judged.verdict == "CORRECT")

    agent_pass_rate = sum(agent_passes) / len(agent_passes)
    fixed_pass_rate = sum(fixed_passes) / len(fixed_passes)

    return RaceRow(
        id=case.id, question=case.question, kind=case.kind,
        agent_answer=agent_result.answer,
        agent_correct="CORRECT" if agent_pass_rate >= 0.5 else "INCORRECT",
        agent_pass_rate=agent_pass_rate,
        agent_seconds=agent_result.elapsed_seconds, agent_cost_usd=agent_result.cost_usd,
        agent_tool_calls=agent_result.tool_calls, agent_stopped_reason=agent_result.stopped_reason,
        fixed_answer=fixed_result.answer,
        fixed_correct="CORRECT" if fixed_pass_rate >= 0.5 else "INCORRECT",
        fixed_pass_rate=fixed_pass_rate,
        fixed_seconds=fixed_result.elapsed_seconds, fixed_cost_usd=fixed_result.cost_usd,
        trials=repeat,
        trajectory_ok=all(trajectory_oks),
        trajectory_notes=trajectory_notes,
        agent_steps=agent_steps,
    )


def print_report(rows: list[RaceRow]) -> None:
    print(f"\n{'#':<3} {'kind':<14} {'agent':<10} {'fixed':<10} {'agent_s':<9} {'fixed_s':<9} {'agent_$':<9} {'tools':<6}")
    for row in rows:
        rates = f"  (agent {row.agent_pass_rate:.0%}, fixed {row.fixed_pass_rate:.0%} of {row.trials})" if row.trials > 1 else ""
        print(f"{row.id:<3} {row.kind:<14} {row.agent_correct:<10} {row.fixed_correct:<10} "
              f"{row.agent_seconds:<9.2f} {row.fixed_seconds:<9.4f} {row.agent_cost_usd:<9.5f} {row.agent_tool_calls:<6}{rates}")
        if row.agent_correct != "CORRECT" or row.fixed_correct != "CORRECT":
            print(f"      agent : {row.agent_answer!r}")
            print(f"      fixed : {row.fixed_answer!r}")
        if row.agent_correct == "CORRECT" and not row.trajectory_ok:
            print(f"      ** OUTCOME-VS-TRAJECTORY GAP -- right answer, wrong path: {row.trajectory_notes}")

    agent_correct_n = sum(1 for r in rows if r.agent_correct == "CORRECT")
    fixed_correct_n = sum(1 for r in rows if r.fixed_correct == "CORRECT")
    total_agent_time = sum(r.agent_seconds for r in rows)
    total_fixed_time = sum(r.fixed_seconds for r in rows)
    total_agent_cost = sum(r.agent_cost_usd for r in rows)

    print(f"\nCorrectness: agent {agent_correct_n}/{len(rows)}, fixed workflow {fixed_correct_n}/{len(rows)}")
    print(f"Total time:  agent {total_agent_time:.2f}s, fixed workflow {total_fixed_time:.4f}s")
    print(f"Total cost:  agent ${total_agent_cost:.5f}, fixed workflow $0.00000 (no model calls)")


def main() -> None:
    # Model answers can contain non-ASCII currency symbols (e.g. Rs sign);
    # Windows terminals often default to a legacy codepage that can't print
    # them, so force UTF-8 on stdout rather than crash mid-report.
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="Race the agent against the fixed workflow.")
    parser.add_argument("--repeat", type=int, default=1, help="Run each case N times and score by majority vote.")
    args = parser.parse_args()

    rows = []
    for case in RACE_CASES:
        print(f"[{case.id}/{len(RACE_CASES)}] {case.question}")
        rows.append(run_case(case, repeat=args.repeat))

    print_report(rows)
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps([asdict(row) for row in rows], indent=2), encoding="utf-8")
    print(f"\nSaved results to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
