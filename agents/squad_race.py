"""Week 10 race: the single agent vs the kitchen squad, same 12 questions.

    python -m agents.squad_race                      # 1 trial per case
    python -m agents.squad_race --repeat 3           # 3 trials, majority vote on quality
    python -m agents.squad_race --cases 7,10 --variants single,squad,squad_sequential
    python -m agents.squad_race --report             # re-print the saved results

Fairness rules (docs/week10-multi-agent.md, "How the race is fair"):
- Same questions (agents/squad_testset.py), same MCP servers and tools, same
  model, same Week 8 HARDENED policy for every agent loop.
- Same meter: every run, single or squad, is timed and metered by this file,
  with one usage scope (rag_chat/client.py usage_scope) around the whole
  run -- so the squad's tokens include the manager and every specialist.
- Order alternates per trial (single first, then squad first), so neither
  side always runs into a rate limit the other one caused.
- Rate-limit sleeps (Groq free tier) are recorded per run and reported, not
  hidden inside "speed": a run that slept is flagged, and latency is
  reported with and without those runs.
- Quality is graded twice: Week 8's deterministic checks (primary -- cheap,
  repeatable) and Week 6's LLM judge (second opinion, catches a right
  number attached to the wrong shop). The judge's own calls are outside
  the measured scope, so grading never counts as either side's cost.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from statistics import mean, median

from agents.agent import run_agent
from agents.mcp_host import McpHost, load_config
from agents.squad import run_squad
from agents.squad_testset import SQUAD_CASES
from agents.testset import RaceCase
from agents.trajectory_eval import check_outcome
from evals.judge import judge_answer
from rag_chat.client import usage_scope

RESULTS_PATH = Path(__file__).resolve().parent / "results" / "squad_race.json"


@dataclass
class Run:
    """One variant answering one question once, with everything measured."""

    answer: str
    checks_pass: bool
    check_reasons: list[str]
    judge: str                 # CORRECT / INCORRECT
    judge_reason: str
    seconds: float
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float
    llm_calls: int
    tool_calls: list[str]
    stopped_reason: str
    rate_limit_waits: int = 0
    rate_limit_wait_seconds: float = 0.0
    detail: dict = field(default_factory=dict)  # squad: plan, manager calls, per-specialist tokens

    @property
    def tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    @property
    def judge_ok(self) -> bool:
        return self.judge == "CORRECT"

    @property
    def correct(self) -> bool:
        """Both graders agree it's right -- the strict reading used for the verdict."""
        return self.checks_pass and self.judge_ok


@dataclass
class CaseResult:
    id: int
    question: str
    kind: str
    ground_truth: str
    runs: dict[str, list[Run]] = field(default_factory=dict)  # variant -> one Run per trial


# -- the variants ---------------------------------------------------------

def _single(question: str, host: McpHost) -> tuple[str, list[str], str, dict]:
    result = run_agent(question, system_prompt=host.system_prompt,
                       tool_schemas=host.tool_schemas, tool_functions=host.tool_functions)
    return result.answer, [s.tool for s in result.steps if s.tool], result.stopped_reason, {}


def _squad(question: str, host: McpHost, parallel: bool = True) -> tuple[str, list[str], str, dict]:
    result = run_squad(question, host, parallel=parallel)
    detail = {
        "plan": result.plan,
        "manager_llm_calls": result.manager_llm_calls,
        "specialists": {
            a.specialist: {"task": a.task, "report": a.result.answer, "llm_calls": a.result.llm_calls,
                           "tokens": a.result.prompt_tokens + a.result.completion_tokens,
                           "seconds": round(a.result.elapsed_seconds, 2)}
            for a in result.assignments
        },
    }
    return result.answer, [s.tool for s in result.steps if s.tool], result.stopped_reason, detail


VARIANTS: dict[str, Callable[[str, McpHost], tuple[str, list[str], str, dict]]] = {
    "single": _single,
    "squad": _squad,
    # Same team, specialists one after another: isolates what parallelism buys.
    "squad_sequential": lambda question, host: _squad(question, host, parallel=False),
}


def measure(variant: str, case: RaceCase, host: McpHost) -> Run:
    """Run one variant on one case inside its own usage scope, then grade it."""
    start = time.perf_counter()
    with usage_scope() as usage:
        answer, tools, stopped, detail = VARIANTS[variant](case.question, host)
    seconds = time.perf_counter() - start
    checks_pass, reasons = check_outcome(answer, case)
    judged = judge_answer(case.question, case.ground_truth, answer)  # outside the scope: not billed to either side
    return Run(
        answer=answer, checks_pass=checks_pass, check_reasons=reasons,
        judge=judged.verdict, judge_reason=judged.reason, seconds=seconds,
        prompt_tokens=usage.prompt_tokens, completion_tokens=usage.completion_tokens,
        cost_usd=usage.cost_usd, llm_calls=usage.calls, tool_calls=tools, stopped_reason=stopped,
        rate_limit_waits=usage.rate_limit_waits, rate_limit_wait_seconds=usage.rate_limit_wait_seconds,
        detail=detail,
    )


def run_race(cases: list[RaceCase], variants: list[str], host: McpHost, repeat: int = 1,
             pause: float = 0.0, log: Callable[[str], None] = print,
             on_case_done: Callable[[list[CaseResult]], None] | None = None,
             existing: list[CaseResult] | None = None) -> list[CaseResult]:
    """Every variant on every case, `repeat` times; `on_case_done` gets all results so far.

    `existing` (from --resume) keeps finished runs: a variant that already
    has `repeat` runs for a case is skipped, so a daily token limit only
    costs the runs that hadn't happened yet.
    """
    rows = {row.id: row for row in existing or []}
    for case in cases:
        row = rows.setdefault(case.id, CaseResult(id=case.id, question=case.question, kind=case.kind,
                                                   ground_truth=case.ground_truth))
        for variant in variants:
            row.runs.setdefault(variant, [])
        todo = [v for v in variants if len(row.runs[v]) < repeat]
        for trial in range(repeat):
            order = todo if trial % 2 == 0 else list(reversed(todo))
            for variant in order:
                if len(row.runs[variant]) > trial:
                    continue
                run = measure(variant, case, host)
                row.runs[variant].append(run)
                flag = " [rate-limited]" if run.rate_limit_waits else ""
                log(f"  #{case.id} {variant:<17} trial {trial + 1}: checks={'PASS' if run.checks_pass else 'FAIL'} "
                    f"judge={run.judge:<9} {run.seconds:5.1f}s {run.tokens:6d} tok ${run.cost_usd:.5f}{flag}")
                if pause:
                    time.sleep(pause)
        if on_case_done and todo:
            on_case_done(sorted(rows.values(), key=lambda r: r.id))  # a later daily-limit error can't lose finished cases
    return sorted(rows.values(), key=lambda r: r.id)


# -- summarising ------------------------------------------------------------

def majority(runs: list[Run], attr: str) -> bool:
    return sum(bool(getattr(r, attr)) for r in runs) * 2 > len(runs)


def summarize(results: list[CaseResult], variant: str, kinds: set[str] | None = None) -> dict:
    """Quality (majority vote per case), speed, tokens and cost for one variant."""
    rows = [r for r in results if variant in r.runs and (kinds is None or r.kind in kinds)]
    runs = [run for r in rows for run in r.runs[variant]]
    if not runs:
        return {}
    clean = [run for run in runs if not run.rate_limit_waits]
    return {
        "cases": len(rows),
        "runs": len(runs),
        "quality_correct": sum(majority(r.runs[variant], "correct") for r in rows),
        "quality_checks": sum(majority(r.runs[variant], "checks_pass") for r in rows),
        "quality_judge": sum(majority(r.runs[variant], "judge_ok") for r in rows),
        "median_seconds": median(run.seconds for run in runs),
        "mean_seconds": mean(run.seconds for run in runs),
        "median_seconds_no_rate_limit": median(run.seconds for run in clean) if clean else None,
        "rate_limited_runs": len(runs) - len(clean),
        "mean_tokens": mean(run.tokens for run in runs),
        "mean_llm_calls": mean(run.llm_calls for run in runs),
        "mean_cost_usd": mean(run.cost_usd for run in runs),
        "total_cost_usd": sum(run.cost_usd for run in runs),
    }


def print_report(results: list[CaseResult], variants: list[str]) -> None:
    trials = max(len(runs) for r in results for runs in r.runs.values())
    print(f"\nPer question ({trials} trial(s) each; quality = majority of trials passing BOTH graders)")
    header = f"{'#':<3} {'kind':<11}" + "".join(f" {v[:16]:>16}" for v in variants)
    print(header)
    for row in results:
        cells = []
        for v in variants:
            runs = row.runs.get(v, [])
            ok = "OK " if majority(runs, "correct") else "BAD"
            cells.append(f" {ok} {median(r.seconds for r in runs):4.1f}s {mean(r.tokens for r in runs):5.0f}t")
        print(f"{row.id:<3} {row.kind:<11}" + "".join(f" {c:>16}" for c in cells))
        for v in variants:
            for i, run in enumerate(row.runs.get(v, [])):
                if not run.correct:
                    why = "; ".join(run.check_reasons) or f"judge: {run.judge_reason}"
                    print(f"      {v} trial {i + 1}: {why}")

    groups = [("ALL", None), ("menu", {"menu"}), ("ingredient", {"ingredient"}),
              ("combined", {"combined"}), ("not_found", {"not_found"})]
    for title, kinds in groups:
        stats = {v: summarize(results, v, kinds) for v in variants}
        if not any(stats.values()):
            continue
        n = next(s["cases"] for s in stats.values() if s)
        print(f"\n== {title} ({n} questions) ==")
        print(f"{'':<26}" + "".join(f"{v:>18}" for v in variants))
        lines = [
            ("Quality: correct (both)", lambda s: f"{s['quality_correct']}/{s['cases']}"),
            ("  deterministic checks", lambda s: f"{s['quality_checks']}/{s['cases']}"),
            ("  LLM judge", lambda s: f"{s['quality_judge']}/{s['cases']}"),
            ("Speed: median s", lambda s: f"{s['median_seconds']:.2f}"),
            ("  median s, no rate-limit", lambda s: "-" if s['median_seconds_no_rate_limit'] is None
                else f"{s['median_seconds_no_rate_limit']:.2f}"),
            ("  rate-limited runs", lambda s: f"{s['rate_limited_runs']}/{s['runs']}"),
            ("Tokens: mean per question", lambda s: f"{s['mean_tokens']:.0f}"),
            ("  LLM calls per question", lambda s: f"{s['mean_llm_calls']:.1f}"),
            ("Cost: mean per question $", lambda s: f"{s['mean_cost_usd']:.5f}"),
            ("  total $", lambda s: f"{s['total_cost_usd']:.5f}"),
        ]
        for label, fmt in lines:
            print(f"{label:<26}" + "".join(f"{fmt(stats[v]) if stats[v] else '-':>18}" for v in variants))
        if "single" in stats and "squad" in stats and stats["single"] and stats["squad"]:
            s, t = stats["single"], stats["squad"]
            print(f"{'squad / single':<26}  tokens x{t['mean_tokens'] / s['mean_tokens']:.2f}, "
                  f"cost x{t['mean_cost_usd'] / max(s['mean_cost_usd'], 1e-12):.2f}, "
                  f"median time x{t['median_seconds'] / s['median_seconds']:.2f}")


def regrade(results: list[CaseResult]) -> None:
    """Re-apply the current deterministic checks to saved answers -- free, for grader fixes.

    Only the rule-based grade changes; the LLM judge's saved verdict is kept.
    """
    cases = {case.id: case for case in SQUAD_CASES}
    for row in results:
        for runs in row.runs.values():
            for run in runs:
                run.checks_pass, run.check_reasons = check_outcome(run.answer, cases[row.id])


def save(results: list[CaseResult], path: Path = RESULTS_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([asdict(r) for r in results], indent=2, ensure_ascii=False), encoding="utf-8")


def load(path: Path = RESULTS_PATH) -> list[CaseResult]:
    rows = []
    for raw in json.loads(path.read_text(encoding="utf-8")):
        runs = {v: [Run(**run) for run in trial_runs] for v, trial_runs in raw.pop("runs").items()}
        rows.append(CaseResult(**raw, runs=runs))
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Race the single agent against the kitchen squad.")
    parser.add_argument("--repeat", type=int, default=1, help="Trials per case (majority vote on quality).")
    parser.add_argument("--cases", help="Comma-separated case ids (default: all 12).")
    parser.add_argument("--variants", default="single,squad", help=f"Comma-separated, from {list(VARIANTS)}.")
    parser.add_argument("--pause", type=float, default=0.0, help="Seconds to wait between runs (rate limits).")
    parser.add_argument("--out", default=str(RESULTS_PATH), help="Where to save results JSON.")
    parser.add_argument("--report", action="store_true", help="Re-print the report from --out and exit.")
    parser.add_argument("--regrade", action="store_true",
                        help="Re-run the deterministic checks on the saved answers in --out (no model calls), save, report.")
    parser.add_argument("--resume", action="store_true",
                        help="Keep finished cases in --out and run only the missing ones (e.g. after a daily token limit).")
    args = parser.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    unknown = set(variants) - set(VARIANTS)
    if unknown:
        parser.error(f"unknown variant(s) {sorted(unknown)}; choose from {list(VARIANTS)}")
    out = Path(args.out)
    if args.regrade:
        results = load(out)
        regrade(results)
        save(results, out)
    if args.report or args.regrade:
        results = load(out)
        print_report(results, [v for v in VARIANTS if any(v in r.runs for r in results)])
        return 0

    cases = SQUAD_CASES
    if args.cases:
        wanted = {int(x) for x in args.cases.split(",")}
        cases = [c for c in SQUAD_CASES if c.id in wanted]

    existing: list[CaseResult] = []
    if args.resume and out.exists():
        existing = load(out)
        missing = sum(max(0, args.repeat - len(r.runs.get(v, []))) for r in existing for v in variants)
        missing += args.repeat * len(variants) * len({c.id for c in cases} - {r.id for r in existing})
        print(f"Resuming from {out}: {missing} run(s) still to do.")

    with McpHost(load_config()) as host:
        if host.failures:
            print(f"MCP servers failed to connect: {host.failures}", file=sys.stderr)
            return 1
        print(f"Racing {variants} on {len(cases)} question(s), {args.repeat} trial(s) each.")
        results = run_race(cases, variants, host, repeat=args.repeat, pause=args.pause,
                           on_case_done=lambda so_far: save(so_far, out), existing=existing)

    print_report(results, variants)
    save(results, out)
    print(f"\nSaved results to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
