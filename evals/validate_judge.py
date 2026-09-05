"""Check the LLM judge agrees with a human before trusting its verdicts.

Reuses the human grading already done by hand in Week 5
(docs/week5-error-analysis.md's Verdict column, carried into
`evals.testset.TestCase.human_verdict`) against the real answers the app
gave at the time (`scripts/week5-traces.json`), instead of re-grading a
fresh sample. Run this once before trusting the judge in `evals/run.py`.

Usage: python -m evals.validate_judge
"""
from __future__ import annotations

import json
from pathlib import Path

from evals.judge import judge_answer
from evals.testset import TEST_CASES

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRACES_PATH = PROJECT_ROOT / "scripts" / "week5-traces.json"


def main() -> None:
    traces_by_number = {trace["number"]: trace for trace in json.loads(TRACES_PATH.read_text(encoding="utf-8"))}

    agreements = 0
    disagreements: list[str] = []
    for case in TEST_CASES:
        trace = traces_by_number.get(case.id)
        if trace is None:
            continue
        judged = judge_answer(case.question, case.ground_truth, trace["answer"])
        judge_says_correct = judged.verdict == "CORRECT"
        human_says_correct = case.human_verdict == "correct"

        if judge_says_correct == human_says_correct:
            agreements += 1
        else:
            disagreements.append(
                f"  #{case.id} {case.question!r}\n"
                f"    human: {case.human_verdict} | judge: {judged.verdict} ({judged.reason})\n"
                f"    app answer was: {trace['answer']!r}"
            )

    total = len(traces_by_number)
    print(f"Judge/human agreement: {agreements}/{total} ({agreements / total:.0%})\n")
    if disagreements:
        print("Disagreements:")
        print("\n".join(disagreements))
    else:
        print("No disagreements.")


if __name__ == "__main__":
    main()
