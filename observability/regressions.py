"""Week 11: the failure-to-test loop's live half -- real failing requests, re-asked.

tests/test_week11_failures.py pins each failure down in code with no API
key. This file keeps the *questions* that failed in production and re-asks
them through the real app (agents/kitchen.py), so a prompt, model or tool
change that brings a failure back shows up before a customer sees it:

    python -m observability.regressions            # every case, live
    python -m observability.regressions --pause 15 # Groq free tier: stay under the per-minute limit

Every case must: finish with outcome "done" (not max_steps / validation
failure), use at most MAX_LLM_CALLS model calls, and make no allergen claim
the ingredient database contradicts. Add a case with
`python -m observability.logs promote <request_id> "why it failed"`.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from rag_chat.config import PROJECT_ROOT

REGRESSIONS_PATH = PROJECT_ROOT / "observability" / "regressions.jsonl"
MAX_LLM_CALLS = 4


@dataclass
class Check:
    case: dict
    request_id: str
    passed: bool
    problems: list[str]
    llm_calls: int
    cost_usd: float
    seconds: float


def load_cases(path: Path = REGRESSIONS_PATH) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def add_case(trace: dict, reason: str, path: Path = REGRESSIONS_PATH) -> dict:
    """Append a logged request's question as a regression case (the data flywheel)."""
    existing = load_cases(path)
    if any(case["question"] == trace["question"] for case in existing):
        raise ValueError("that question is already a regression case")
    case = {"source_request_id": trace["request_id"], "logged_at": trace["timestamp"],
            "question": trace["question"], "failure": reason,
            "failed_outcome": trace.get("outcome"), "failed_answer": trace.get("answer")}
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(case, ensure_ascii=False) + "\n")
    return case


def judge(case: dict, trace: dict) -> list[str]:
    """Why a re-asked case still fails; empty means it passed."""
    from observability.logs import audit

    problems = []
    if trace.get("outcome") != "done":
        problems.append(f"outcome {trace.get('outcome')}")
    if trace.get("llm_calls", 0) > MAX_LLM_CALLS:
        problems.append(f"{trace['llm_calls']} LLM calls (max {MAX_LLM_CALLS})")
    for finding in audit([trace]):
        problems.append(f"says {finding['dish']} is {finding['allergen']}-free")
    return problems


def run(cases: list[dict], *, pause: float = 0.0) -> list[Check]:
    from agents.kitchen import ask
    from agents.mcp_host import DEFAULT_CONFIG, McpHost, load_config

    checks = []
    with McpHost(load_config(DEFAULT_CONFIG)) as host:
        for index, case in enumerate(cases):
            if index and pause:
                time.sleep(pause)
            reply = ask(case["question"], host, user_id="regression", session_id="regressions")
            trace = reply.trace.to_json()
            problems = judge(case, trace)
            checks.append(Check(case=case, request_id=reply.request_id, passed=not problems, problems=problems,
                                llm_calls=trace["llm_calls"], cost_usd=trace["cost_usd"],
                                seconds=trace["duration_ms"] / 1000))
    return checks


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Re-ask every production failure through the live app.")
    parser.add_argument("--pause", type=float, default=0.0, help="Seconds between cases (rate limits).")
    args = parser.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    cases = load_cases()
    if not cases:
        print("No regression cases yet.")
        return 0
    checks = run(cases, pause=args.pause)
    for check in checks:
        status = "PASS" if check.passed else "FAIL"
        print(f"{status}  {check.request_id}  {check.llm_calls} calls  {check.seconds:5.1f}s  ${check.cost_usd:.5f}  "
              f"{check.case['question'][:60]}  (was: {check.case['failed_outcome']}, {check.case['source_request_id']})")
        for problem in check.problems:
            print(f"      {problem}")
    failed = sum(not c.passed for c in checks)
    print(f"\n{len(checks) - failed}/{len(checks)} regression cases pass")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
