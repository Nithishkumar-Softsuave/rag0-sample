"""Week 8 batch evaluation: run every case under a policy, many times, and
measure the whole path -- not just the final answer.

Reports every metric the Week 8 brief lists:
- outcome pass rate (deterministic checks, no LLM judge)
- trajectory pass rate + tool-choice accuracy (exact match, precision, recall)
- the outcome-vs-trajectory gap: right answer via a wrong path, and vice versa
- failure-mode counts (agents/failure_modes.py)
- cost per task, mean and p99 -- actual, plus an estimate at gpt-4o-mini list
  prices, since the Groq free tier this was verified on bills $0
- tokens and latency per task, mean and p99

`--policy baseline` reproduces Week 7 exactly (frozen prompt + tools from
agents/baseline.py, every defense off); `--policy hardened` is what the app
runs. `--policy both` runs both and prints the before/after table.

Usage:
    python -m agents.trajectory_eval                        # both policies, all 16 cases, 3 trials
    python -m agents.trajectory_eval --policy hardened --cases adversarial --repeat 1
    python -m agents.trajectory_eval --compare              # re-print the table from saved results
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from statistics import mean

from agents.adversarial_testset import ADVERSARIAL_CASES
from agents.agent import AGENT_SYSTEM_PROMPT, BASELINE, HARDENED, run_agent
from agents.baseline import WEEK7_AGENT_SYSTEM_PROMPT, WEEK7_TOOL_FUNCTIONS
from agents.failure_modes import ALL_MODES, classify_run
from agents.testset import RACE_CASES, RaceCase
from agents.tools import TOOL_FUNCTIONS
from agents.trajectory import check_trajectory, tool_choice_scores
from rag_chat.client import PRICING_PER_MILLION_TOKENS

RESULTS_DIR = Path(__file__).resolve().parent / "results"
REFERENCE_PRICING = PRICING_PER_MILLION_TOKENS["openai/gpt-4o-mini"]

POLICIES = {
    "baseline": {"system_prompt": WEEK7_AGENT_SYSTEM_PROMPT, "tool_functions": WEEK7_TOOL_FUNCTIONS, "policy": BASELINE},
    "hardened": {"system_prompt": AGENT_SYSTEM_PROMPT, "tool_functions": TOOL_FUNCTIONS, "policy": HARDENED},
}


def _normalize(text: str) -> str:
    """Lowercase; drop thousands separators, markdown emphasis, curly quotes and odd
    spaces -- so "₹ 2,074" matches "2074" and "**not** pricier" / "can’t" match plainly.

    Markdown and curly apostrophes were added after reading the first full
    run by hand: correct answers like "the meal is **not** pricier" and "I
    can’t provide the USD amount" were graded as failures.
    """
    text = text.lower().replace(" ", " ").replace(" ", " ")
    text = text.replace("’", "'").replace("‘", "'")
    text = re.sub(r"[*_`]", "", text)
    return re.sub(r"(?<=\d),(?=\d)", "", text)


def check_outcome(answer: str, case: RaceCase) -> tuple[bool, list[str]]:
    """Grade an answer with the case's deterministic checks. Returns (passed, reasons)."""
    text = _normalize(answer)
    reasons: list[str] = []
    for group in case.must_include:
        if not any(_normalize(phrase) in text for phrase in group):
            reasons.append(f"missing any of {list(group)[:4]}{'...' if len(group) > 4 else ''}")
    for phrase in case.must_not_include:
        if _normalize(phrase) in text:
            reasons.append(f"contains forbidden {phrase!r}")
    if case.ordered:
        positions = [text.find(_normalize(phrase)) for phrase in case.ordered]
        if -1 in positions or positions != sorted(positions):
            reasons.append(f"not in order {list(case.ordered)}")
    return not reasons, reasons


def _estimated_cost(prompt_tokens: int, completion_tokens: int) -> float:
    return (prompt_tokens * REFERENCE_PRICING["input"] + completion_tokens * REFERENCE_PRICING["output"]) / 1_000_000


def p99(values: list[float]) -> float:
    """Nearest-rank 99th percentile. With n < 100 this is simply the worst run."""
    ordered = sorted(values)
    return ordered[max(0, math.ceil(0.99 * len(ordered)) - 1)] if ordered else 0.0


def run_one(case: RaceCase, policy_name: str, trial: int) -> dict:
    config = POLICIES[policy_name]
    result = run_agent(case.question, **config)
    outcome_ok, outcome_reasons = check_outcome(result.answer, case)
    trajectory = check_trajectory(result.steps, list(case.expected_tools))
    precision, recall = tool_choice_scores(result.steps, list(case.expected_tools))
    return {
        "case_id": case.id, "kind": case.kind, "targets": case.targets, "trial": trial,
        "question": case.question, "answer": result.answer,
        "outcome_ok": outcome_ok, "outcome_reasons": outcome_reasons,
        "trajectory_ok": trajectory.ok, "trajectory_note": trajectory.note,
        "tool_precision": precision, "tool_recall": recall,
        "failure_modes": classify_run(result, case.expected_tools, config["system_prompt"]),
        "stopped_reason": result.stopped_reason, "tool_calls": result.tool_calls,
        "blocked_calls": sum(1 for step in result.steps if step.blocked),
        "validation_issues": result.validation_issues, "validation_retries": result.validation_retries,
        "prompt_tokens": result.prompt_tokens, "completion_tokens": result.completion_tokens,
        "llm_calls": result.llm_calls, "cost_usd": result.cost_usd,
        "est_cost_usd_gpt4o_mini": _estimated_cost(result.prompt_tokens, result.completion_tokens),
        "seconds": result.elapsed_seconds,
        "steps": [asdict(step) for step in result.steps],
    }


def summarize(runs: list[dict]) -> dict:
    n = len(runs)
    tokens = [r["prompt_tokens"] + r["completion_tokens"] for r in runs]
    per_case: dict[int, dict] = {}
    for case_id in sorted({r["case_id"] for r in runs}):
        rows = [r for r in runs if r["case_id"] == case_id]
        modes = Counter(mode for r in rows for mode in r["failure_modes"])
        per_case[case_id] = {
            "kind": rows[0]["kind"], "trials": len(rows),
            "outcome_pass_rate": sum(r["outcome_ok"] for r in rows) / len(rows),
            "trajectory_pass_rate": sum(r["trajectory_ok"] for r in rows) / len(rows),
            "failure_modes": dict(modes),
        }
    return {
        "runs": n,
        "outcome_pass_rate": sum(r["outcome_ok"] for r in runs) / n,
        "trajectory_pass_rate": sum(r["trajectory_ok"] for r in runs) / n,
        "tool_choice_precision": mean(r["tool_precision"] for r in runs),
        "tool_choice_recall": mean(r["tool_recall"] for r in runs),
        "gap_right_answer_wrong_path": sum(r["outcome_ok"] and not r["trajectory_ok"] for r in runs),
        "gap_right_path_wrong_answer": sum(r["trajectory_ok"] and not r["outcome_ok"] for r in runs),
        "runs_with_any_failure_mode": sum(bool(r["failure_modes"]) for r in runs),
        "failure_mode_counts": {mode: sum(mode in r["failure_modes"] for r in runs) for mode in ALL_MODES},
        "blocked_calls": sum(r["blocked_calls"] for r in runs),
        "validation_retries": sum(r["validation_retries"] for r in runs),
        "failed_closed": sum(r["stopped_reason"] == "validation_failed" for r in runs),
        "cost_usd_mean": mean(r["cost_usd"] for r in runs), "cost_usd_p99": p99([r["cost_usd"] for r in runs]),
        "est_cost_usd_mean": mean(r["est_cost_usd_gpt4o_mini"] for r in runs),
        "est_cost_usd_p99": p99([r["est_cost_usd_gpt4o_mini"] for r in runs]),
        "tokens_mean": mean(tokens), "tokens_p99": p99(tokens),
        "seconds_mean": mean(r["seconds"] for r in runs), "seconds_p99": p99([r["seconds"] for r in runs]),
        "llm_calls_mean": mean(r["llm_calls"] for r in runs),
        "per_case": per_case,
    }


def print_summary(policy_name: str, summary: dict) -> None:
    print(f"\n=== {policy_name}: {summary['runs']} runs ===")
    print(f"{'case':<6}{'kind':<24}{'outcome':<10}{'path':<8}failure modes")
    for case_id, row in summary["per_case"].items():
        modes = ", ".join(f"{m} x{c}" for m, c in row["failure_modes"].items()) or "-"
        print(f"{case_id:<6}{row['kind']:<24}{row['outcome_pass_rate']:<10.0%}{row['trajectory_pass_rate']:<8.0%}{modes}")


def _row(label: str, before, after, fmt: str) -> str:
    return f"{label:<42}{format(before, fmt):>14}{format(after, fmt):>14}"


def print_comparison(before: dict, after: dict) -> None:
    print(f"\n{'metric':<42}{'baseline':>14}{'hardened':>14}")
    print(_row("outcome pass rate", before["outcome_pass_rate"], after["outcome_pass_rate"], ".0%"))
    print(_row("trajectory pass rate (tool-choice exact)", before["trajectory_pass_rate"], after["trajectory_pass_rate"], ".0%"))
    print(_row("tool-choice precision", before["tool_choice_precision"], after["tool_choice_precision"], ".2f"))
    print(_row("tool-choice recall", before["tool_choice_recall"], after["tool_choice_recall"], ".2f"))
    print(_row("right answer, wrong path (runs)", before["gap_right_answer_wrong_path"], after["gap_right_answer_wrong_path"], "d"))
    print(_row("right path, wrong answer (runs)", before["gap_right_path_wrong_answer"], after["gap_right_path_wrong_answer"], "d"))
    print(_row("runs with any failure mode", before["runs_with_any_failure_mode"], after["runs_with_any_failure_mode"], "d"))
    for mode in ALL_MODES:
        print(_row(f"  {mode}", before["failure_mode_counts"][mode], after["failure_mode_counts"][mode], "d"))
    print(_row("tool calls blocked by guards", before["blocked_calls"], after["blocked_calls"], "d"))
    print(_row("answers failed closed by validation", before["failed_closed"], after["failed_closed"], "d"))
    print(_row("cost/task mean, actual ($)", before["cost_usd_mean"], after["cost_usd_mean"], ".5f"))
    print(_row("cost/task p99, actual ($)", before["cost_usd_p99"], after["cost_usd_p99"], ".5f"))
    print(_row("cost/task mean, est. gpt-4o-mini ($)", before["est_cost_usd_mean"], after["est_cost_usd_mean"], ".5f"))
    print(_row("cost/task p99, est. gpt-4o-mini ($)", before["est_cost_usd_p99"], after["est_cost_usd_p99"], ".5f"))
    print(_row("tokens/task mean", before["tokens_mean"], after["tokens_mean"], ".0f"))
    print(_row("tokens/task p99", before["tokens_p99"], after["tokens_p99"], ".0f"))
    print(_row("seconds/task mean", before["seconds_mean"], after["seconds_mean"], ".1f"))
    print(_row("seconds/task p99", before["seconds_p99"], after["seconds_p99"], ".1f"))


def _results_path(policy_name: str) -> Path:
    return RESULTS_DIR / f"week8_eval_{policy_name}.json"


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="Week 8 trajectory evaluation.")
    parser.add_argument("--policy", choices=["baseline", "hardened", "both"], default="both")
    parser.add_argument("--cases", choices=["official", "adversarial", "all"], default="all")
    parser.add_argument("--ids", type=int, nargs="*", help="Only run these case ids.")
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--compare", action="store_true", help="Only re-print the before/after table from saved results.")
    parser.add_argument("--regrade", action="store_true",
                        help="Re-score saved answers with the current outcome checks (no agent calls), then compare.")
    args = parser.parse_args()

    if args.regrade:
        cases_by_id = {case.id: case for case in RACE_CASES + ADVERSARIAL_CASES}
        for policy_name in ("baseline", "hardened"):
            path = _results_path(policy_name)
            if not path.exists():
                continue
            runs = json.loads(path.read_text(encoding="utf-8"))["runs"]
            for run in runs:
                run["outcome_ok"], run["outcome_reasons"] = check_outcome(run["answer"], cases_by_id[run["case_id"]])
            summary = summarize(runs)
            print_summary(policy_name, summary)
            path.write_text(json.dumps({"summary": summary, "runs": runs}, indent=2, ensure_ascii=False), encoding="utf-8")
    elif not args.compare:
        cases = {"official": RACE_CASES, "adversarial": ADVERSARIAL_CASES, "all": RACE_CASES + ADVERSARIAL_CASES}[args.cases]
        if args.ids:
            cases = [case for case in cases if case.id in args.ids]
        policy_names = ["baseline", "hardened"] if args.policy == "both" else [args.policy]
        for policy_name in policy_names:
            runs = []
            for case in cases:
                for trial in range(args.repeat):
                    print(f"[{policy_name}] case {case.id} trial {trial + 1}/{args.repeat}", flush=True)
                    runs.append(run_one(case, policy_name, trial))
            summary = summarize(runs)
            print_summary(policy_name, summary)
            RESULTS_DIR.mkdir(parents=True, exist_ok=True)
            _results_path(policy_name).write_text(
                json.dumps({"summary": summary, "runs": runs}, indent=2, ensure_ascii=False), encoding="utf-8",
            )

    saved = {name: _results_path(name) for name in ("baseline", "hardened")}
    if all(path.exists() for path in saved.values()):
        before = json.loads(saved["baseline"].read_text(encoding="utf-8"))["summary"]
        after = json.loads(saved["hardened"].read_text(encoding="utf-8"))["summary"]
        print_comparison(before, after)


if __name__ == "__main__":
    main()
