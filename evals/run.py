"""The one-command Week 6 test set: runs every case in evals/testset.py
through the real retrieval + generation pipeline, scores it with the free
rule checks first and the LLM judge second, and prints a score overall and
per problem group (matching Week 5's named failure groups A-E).

Chat-model answers vary run to run (temperature > 0), so a single pass can
make a fix look better or worse than it really is by luck -- Week 4's
analysis handled this by re-running its adversarial queries 3x before
trusting a verdict (docs/week4-wrong-query-findings.md). `--repeat N` does
the same here: each case runs N times and is scored by majority vote, with
the per-trial pass rate kept alongside so the noise itself is visible
instead of hidden behind one lucky/unlucky number.

Usage:
    python -m evals.run                     # run once, print + save results
    python -m evals.run --repeat 3           # run each case 3x, majority vote
    python -m evals.run --label before       # save under a specific label
    python -m evals.run --compare before after   # diff two saved runs
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from evals.checks import check_refusal_behavior, check_sources
from evals.judge import judge_answer
from evals.testset import FAILURE_GROUP_NAMES, TEST_CASES, TestCase
from rag_chat.config import get_settings
from rag_chat.ingestion import index_directory
from rag_chat.retrieval import generate_response, retrieve

RESULTS_DIR = Path(__file__).resolve().parent / "results"


@dataclass
class CaseResult:
    id: int
    question: str
    group: str  # failure group letter, or "none"
    passed: bool  # majority vote across trials
    pass_rate: float  # fraction of trials that passed -- shows noise `passed` hides
    source_check: bool  # from the majority (or only) trial
    refusal_check: bool
    judge_verdict: str  # "CORRECT" / "INCORRECT" / "SKIPPED" (refusal cases)
    judge_reason: str
    sources: list[str]
    answer: str
    trials: list[dict] = field(default_factory=list)  # every trial's raw outcome


def run_one_trial(case: TestCase) -> dict:
    """One live call through retrieve() + generate_response(), scored once."""
    chunks, sources = retrieve(case.question)
    answer = generate_response(case.question, chunks)

    source_check = check_sources(case.expected_sources, sources)
    refusal_check = check_refusal_behavior(case.should_refuse, answer)

    if case.should_refuse:
        # A rule already covers "did it refuse" -- grading a refusal against
        # a ground-truth sentence adds nothing a human grader would want.
        judge_verdict, judge_reason = "SKIPPED", "Refusal case -- rule check only."
        passed = source_check and refusal_check
    else:
        judged = judge_answer(case.question, case.ground_truth, answer)
        judge_verdict, judge_reason = judged.verdict, judged.reason
        passed = source_check and refusal_check and judged.verdict == "CORRECT"

    return {
        "passed": passed,
        "source_check": source_check,
        "refusal_check": refusal_check,
        "judge_verdict": judge_verdict,
        "judge_reason": judge_reason,
        "sources": sources,
        "answer": answer,
    }


def run_case(case: TestCase, repeat: int = 1) -> CaseResult:
    """Run one test case `repeat` times and score it by majority vote."""
    trials = [run_one_trial(case) for _ in range(repeat)]
    pass_rate = sum(1 for trial in trials if trial["passed"]) / len(trials)
    # The trial closest to the majority verdict is shown as "the" answer --
    # arbitrary among ties, but representative rather than always trial 0.
    majority = pass_rate >= 0.5
    representative = next(trial for trial in trials if trial["passed"] == majority)

    return CaseResult(
        id=case.id,
        question=case.question,
        group=case.failure_group or "none",
        passed=majority,
        pass_rate=pass_rate,
        source_check=representative["source_check"],
        refusal_check=representative["refusal_check"],
        judge_verdict=representative["judge_verdict"],
        judge_reason=representative["judge_reason"],
        sources=representative["sources"],
        answer=representative["answer"],
        trials=trials,
    )


def score_by_group(results: list[CaseResult]) -> dict[str, tuple[int, int]]:
    """Return {group: (passed, total)} for every group present in the results."""
    scores: dict[str, tuple[int, int]] = {}
    for group in sorted({result.group for result in results}):
        in_group = [result for result in results if result.group == group]
        scores[group] = (sum(1 for result in in_group if result.passed), len(in_group))
    return scores


def print_report(results: list[CaseResult]) -> None:
    total_passed = sum(1 for result in results if result.passed)
    print(f"\nOverall: {total_passed}/{len(results)} ({total_passed / len(results):.0%})\n")
    print("By problem group:")
    for group, (passed, total) in score_by_group(results).items():
        name = FAILURE_GROUP_NAMES.get(group, "Previously passing (no known failure)")
        print(f"  {group:>4} - {name}: {passed}/{total}")
    print()
    for result in results:
        if not result.passed:
            rate = f" pass_rate={result.pass_rate:.0%} (n={len(result.trials)})" if len(result.trials) > 1 else ""
            print(f"  FAIL #{result.id} [{result.group}] {result.question}{rate}")
            print(f"       sources_ok={result.source_check} refusal_ok={result.refusal_check} "
                  f"judge={result.judge_verdict} ({result.judge_reason})")
            print(f"       answer: {result.answer!r}")


def save_results(results: list[CaseResult], label: str) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / f"{label}.json"
    path.write_text(json.dumps([asdict(result) for result in results], indent=2), encoding="utf-8")
    return path


def compare(label_a: str, label_b: str) -> None:
    results_a = {item["id"]: item for item in json.loads((RESULTS_DIR / f"{label_a}.json").read_text(encoding="utf-8"))}
    results_b = {item["id"]: item for item in json.loads((RESULTS_DIR / f"{label_b}.json").read_text(encoding="utf-8"))}

    def score(results: dict) -> dict[str, tuple[int, int]]:
        scores: dict[str, tuple[int, int]] = {}
        for group in sorted({item["group"] for item in results.values()}):
            in_group = [item for item in results.values() if item["group"] == group]
            scores[group] = (sum(1 for item in in_group if item["passed"]), len(in_group))
        return scores

    before, after = score(results_a), score(results_b)
    total_before = (sum(p for p, _ in before.values()), sum(t for _, t in before.values()))
    total_after = (sum(p for p, _ in after.values()), sum(t for _, t in after.values()))

    print(f"\n{label_a} -> {label_b}\n")
    print(f"  Overall: {total_before[0]}/{total_before[1]} -> {total_after[0]}/{total_after[1]}")
    print("\n  By problem group:")
    for group in sorted(set(before) | set(after)):
        name = FAILURE_GROUP_NAMES.get(group, "Previously passing (no known failure)")
        before_score = before.get(group, (0, 0))
        after_score = after.get(group, (0, 0))
        print(f"    {group:>4} - {name}: {before_score[0]}/{before_score[1]} -> {after_score[0]}/{after_score[1]}")

    flipped = [
        item_id for item_id, item_a in results_a.items()
        if item_id in results_b and not item_a["passed"] and results_b[item_id]["passed"]
    ]
    regressed = [
        item_id for item_id, item_a in results_a.items()
        if item_id in results_b and item_a["passed"] and not results_b[item_id]["passed"]
    ]
    print(f"\n  Fixed: {sorted(flipped)}")
    print(f"  Regressed: {sorted(regressed)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Week 6 eval test set.")
    parser.add_argument("--label", default="current", help="Name to save this run's results under.")
    parser.add_argument("--repeat", type=int, default=1, help="Run each case N times and score by majority vote.")
    parser.add_argument("--compare", nargs=2, metavar=("BEFORE", "AFTER"), help="Diff two saved runs instead of running.")
    args = parser.parse_args()

    if args.compare:
        compare(*args.compare)
        return

    indexed = index_directory(get_settings().articles_dir)
    if indexed:
        print(f"Checked {len(indexed)} document(s); only new content was embedded.")

    results = []
    for case in TEST_CASES:
        print(f"[{case.id}/{len(TEST_CASES)}] {case.question}")
        results.append(run_case(case, repeat=args.repeat))

    print_report(results)
    path = save_results(results, args.label)
    print(f"\nSaved results to {path}")


if __name__ == "__main__":
    main()
