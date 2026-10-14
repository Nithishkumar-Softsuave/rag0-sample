"""Week 11: the support-side tools over data/traces.jsonl.

    python -m observability.logs find dairy swap                 # every word must match
    python -m observability.logs find --user u_042 --since 3d
    python -m observability.logs find --outcome validation_failed --route squad
    python -m observability.logs show req_1a2b3c4d5e6f           # one request, step by step
    python -m observability.logs stats --since 1d                # cost & latency per request, per step
    python -m observability.logs audit --allergen dairy --since 7d  # answers that call a dish X-free when it isn't
    python -m observability.logs recheck --since 1d              # would today's output checks reject past answers?
    python -m observability.logs compare - week11-after          # same questions, two runs (by session; '-' = none)

`find` searches the question and the answer (and tool names, with --tool);
`show` prints the span tree with the time and cost of every step, the tool
arguments and results, and the final answer -- enough to explain *why* an
answer came out the way it did, not just what it said.
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

from observability.tracing import TRACE_PATH, read_traces

_RELATIVE = re.compile(r"^(\d+(?:\.\d+)?)([mhdw])$")
_UNITS = {"m": "minutes", "h": "hours", "d": "days", "w": "weeks"}


def parse_when(text: str, now: datetime | None = None) -> datetime:
    """'3d', '12h', '30m', '1w' (ago), or an ISO date/time ('2026-10-12', '2026-10-12T09:30')."""
    now = now or datetime.now(timezone.utc)
    match = _RELATIVE.match(text.strip())
    if match:
        return now - timedelta(**{_UNITS[match.group(2)]: float(match.group(1))})
    when = datetime.fromisoformat(text.strip())
    return when if when.tzinfo else when.replace(tzinfo=timezone.utc)


def _when(trace: dict) -> datetime:
    return datetime.fromisoformat(trace["timestamp"])


def _haystack(trace: dict) -> str:
    return f"{trace.get('question') or ''}\n{trace.get('answer') or ''}".lower()


def filter_traces(traces: list[dict], *, words: list[str] = (), user: str | None = None,
                  session: str | None = None, route: str | None = None, outcome: str | None = None,
                  tool: str | None = None, since: datetime | None = None, until: datetime | None = None,
                  min_cost: float | None = None, min_seconds: float | None = None) -> list[dict]:
    """The traces matching every given filter, newest first."""
    words = [w.lower() for w in words]
    hits = []
    for trace in traces:
        if words and not all(word in _haystack(trace) for word in words):
            continue
        if user and trace.get("user_id") != user:
            continue
        if session and trace.get("session_id") != session:
            continue
        if route and trace.get("route") != route:
            continue
        if outcome and trace.get("outcome") != outcome:
            continue
        if tool and tool not in (trace.get("attrs", {}).get("tool_calls") or []):
            continue
        if since and _when(trace) < since:
            continue
        if until and _when(trace) > until:
            continue
        if min_cost is not None and trace.get("cost_usd", 0) < min_cost:
            continue
        if min_seconds is not None and trace.get("duration_ms", 0) < min_seconds * 1000:
            continue
        hits.append(trace)
    return sorted(hits, key=_when, reverse=True)


def _one_line(text: str | None, width: int) -> str:
    text = " ".join((text or "").split())
    return text if len(text) <= width else text[: width - 1] + "…"


def format_row(trace: dict) -> str:
    when = _when(trace).strftime("%Y-%m-%d %H:%M")
    return (f"{trace['request_id']}  {when}  {(trace.get('user_id') or '-'):<8} {trace.get('route', '-'):<6} "
            f"{(trace.get('outcome') or '-'):<17} {trace.get('duration_ms', 0) / 1000:5.1f}s "
            f"${trace.get('cost_usd', 0):.5f}  {_one_line(trace.get('question'), 70)}")


def format_trace(trace: dict, *, full: bool = False) -> str:
    """One request as a readable span tree: time and cost per step, tool args and results."""
    width = None if full else 220
    attrs = trace.get("attrs", {})
    lines = [
        f"request   {trace['request_id']}   {trace['timestamp']}",
        f"user      {trace.get('user_id') or '-'}   session {trace.get('session_id') or '-'}   route {trace.get('route')}",
        f"model     {attrs.get('model')} via {attrs.get('provider')}   app {attrs.get('app_version')}   "
        f"host prompt {attrs.get('host_prompt_version', '-')}",
        f"outcome   {trace.get('outcome')}   {trace.get('duration_ms', 0) / 1000:.2f}s   "
        f"{trace.get('llm_calls', 0)} LLM calls   {trace.get('prompt_tokens', 0)}+{trace.get('completion_tokens', 0)} tokens   "
        f"${trace.get('cost_usd', 0):.6f}"
        + (f"   (rate-limit wait {trace['rate_limit_wait_seconds']:.0f}s)" if trace.get("rate_limit_wait_seconds") else ""),
        "",
        f"Q: {trace.get('question')}",
        "",
        f"{'step':<44} {'start':>7} {'time':>7} {'tokens':>11} {'cost':>10}",
    ]
    spans = trace.get("spans", [])
    children: dict[str | None, list[dict]] = defaultdict(list)
    for item in spans:
        children[item.get("parent_id")].append(item)

    def walk(parent: str | None, depth: int) -> None:
        for item in sorted(children[parent], key=lambda s: s["start_ms"]):
            tokens = f"{item['prompt_tokens']}+{item['completion_tokens']}" if item["llm_calls"] else ""
            cost = f"${item['cost_usd']:.6f}" if item["llm_calls"] else ""
            label = ("  " * depth + item["name"])[:44]
            lines.append(f"{label:<44} {item['start_ms'] / 1000:6.2f}s {item['duration_ms'] / 1000:6.2f}s "
                         f"{tokens:>11} {cost:>10}" + (f"  ERROR {item['error']}" if item.get("error") else ""))
            pad = "  " * depth + "    "
            a = item.get("attrs", {})
            if item["kind"] == "tool":
                lines.append(pad + "args   " + json.dumps(a.get("args"), ensure_ascii=False))
                if a.get("blocked"):
                    lines.append(pad + "BLOCKED " + str(a["blocked"]))
                result = json.dumps(a.get("result"), ensure_ascii=False)
                lines.append(pad + "result " + (result if width is None else _one_line(result, width)))
            elif item["kind"] == "llm" and (a.get("tool_calls") or a.get("output")):
                said = f"calls {a['tool_calls']}" if a.get("tool_calls") else _one_line(a.get("output"), 160)
                lines.append(pad + said)
            elif item["kind"] == "check" and a.get("issues"):
                lines.append(pad + "ISSUES " + "; ".join(a["issues"]))
            elif item["kind"] == "agent" and item.get("parent_id") and a.get("task") and item["name"].startswith("specialist"):
                lines.append(pad + "task   " + _one_line(a["task"], 160))
            walk(item["id"], depth + 1)

    walk(None, 0)
    lines += ["", f"A: {trace.get('answer')}"]
    if attrs.get("validation_issues"):
        lines.append(f"validation issues: {attrs['validation_issues']}")
    if attrs.get("error"):
        lines.append(f"error: {attrs['error']}")
    return "\n".join(lines)


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round(pct / 100 * (len(ordered) - 1))))
    return ordered[index]


def stats(traces: list[dict]) -> dict:
    """Cost and latency per request, and where the time and money go, step by step."""
    if not traces:
        return {"requests": 0}
    costs = [t.get("cost_usd", 0) for t in traces]
    seconds = [t.get("duration_ms", 0) / 1000 for t in traces]
    tokens = [t.get("prompt_tokens", 0) + t.get("completion_tokens", 0) for t in traces]
    by_step: dict[str, dict[str, float]] = defaultdict(lambda: {"count": 0, "seconds": 0.0, "cost": 0.0, "tokens": 0})
    for trace in traces:
        for item in trace.get("spans", []):
            # Leaf work only, so nested spans aren't counted twice: LLM calls, tools, checks.
            if item["kind"] not in ("llm", "tool", "check"):
                continue
            step = by_step[item["name"]]
            step["count"] += 1
            step["seconds"] += item["duration_ms"] / 1000
            step["cost"] += item["cost_usd"]
            step["tokens"] += item["prompt_tokens"] + item["completion_tokens"]
    outcomes: dict[str, int] = defaultdict(int)
    for trace in traces:
        outcomes[trace.get("outcome") or "-"] += 1
    return {
        "requests": len(traces),
        "cost_per_request_mean": statistics.fmean(costs),
        "cost_per_request_p50": _percentile(costs, 50),
        "cost_per_request_p95": _percentile(costs, 95),
        "cost_total": sum(costs),
        "tokens_per_request_mean": statistics.fmean(tokens),
        "llm_calls_per_request_mean": statistics.fmean(t.get("llm_calls", 0) for t in traces),
        "seconds_p50": _percentile(seconds, 50),
        "seconds_p95": _percentile(seconds, 95),
        "rate_limited_requests": sum(1 for t in traces if t.get("rate_limit_wait_seconds")),
        "outcomes": dict(outcomes),
        "by_step": {name: dict(values) for name, values in sorted(by_step.items(), key=lambda kv: -kv[1]["cost"])},
    }


def reprice(trace: dict) -> dict:
    """A copy of `trace` with every LLM call's cost recomputed from its logged tokens at today's prices.

    Week 11's first live runs were logged before cached prompt tokens were
    priced at Groq's cached rate; repricing puts two runs on the same scale.
    """
    from rag_chat.client import PRICING_PER_MILLION_TOKENS

    copy = json.loads(json.dumps(trace))
    total = 0.0
    for item in copy.get("spans", []):
        if item["kind"] != "llm" or not item["llm_calls"]:
            continue
        model = item["attrs"].get("model") or copy.get("attrs", {}).get("model")
        rates = PRICING_PER_MILLION_TOKENS.get(model, {"input": 0.0, "output": 0.0})
        cached = item.get("cached_prompt_tokens") or item["attrs"].get("cached_tokens") or 0
        item["cost_usd"] = ((item["prompt_tokens"] - cached) * rates["input"]
                            + cached * rates.get("cached_input", rates["input"])
                            + item["completion_tokens"] * rates["output"]) / 1e6
        total += item["cost_usd"]
    copy["cost_usd"] = total
    return copy


def compare(before: list[dict], after: list[dict]) -> dict:
    """Pair two runs' requests by question (latest of each) and total up cost, calls and outcomes."""
    def latest(traces: list[dict]) -> dict[str, dict]:
        by_question: dict[str, dict] = {}
        for trace in sorted(traces, key=_when):
            by_question[trace["question"]] = reprice(trace)
        return by_question

    a, b = latest(before), latest(after)
    pairs = [(a[q], b[q]) for q in a if q in b]

    def totals(side: list[dict]) -> dict:
        n = len(side) or 1
        return {"cost_mean": sum(t["cost_usd"] for t in side) / n,
                "llm_calls_mean": sum(t.get("llm_calls", 0) for t in side) / n,
                "tokens_mean": sum(t.get("prompt_tokens", 0) + t.get("completion_tokens", 0) for t in side) / n,
                "seconds_p50": _percentile([t.get("duration_ms", 0) / 1000 for t in side], 50),
                "seconds_max": max((t.get("duration_ms", 0) / 1000 for t in side), default=0),
                "answered": sum(1 for t in side if t.get("outcome") == "done")}

    return {"pairs": pairs, "before": totals([p[0] for p in pairs]), "after": totals([p[1] for p in pairs])}


def format_compare(result: dict) -> str:
    lines = [f"{'question':<62} {'before':>22} {'after':>22}"]
    for old, new in result["pairs"]:
        cell = lambda t: f"{t.get('outcome', '-')[:9]:<9} {t['llm_calls']}c ${t['cost_usd']:.5f}"
        lines.append(f"{_one_line(old['question'], 62):<62} {cell(old):>22} {cell(new):>22}")
    b, a = result["before"], result["after"]
    n = len(result["pairs"])
    change = (a["cost_mean"] / b["cost_mean"] - 1) * 100 if b["cost_mean"] else 0.0
    lines += ["",
              f"{n} questions in both runs",
              f"cost per request   ${b['cost_mean']:.6f} -> ${a['cost_mean']:.6f}  ({change:+.0f}%)",
              f"LLM calls/request  {b['llm_calls_mean']:.2f} -> {a['llm_calls_mean']:.2f}",
              f"tokens/request     {b['tokens_mean']:.0f} -> {a['tokens_mean']:.0f}",
              f"latency p50 / max  {b['seconds_p50']:.1f}s / {b['seconds_max']:.1f}s -> {a['seconds_p50']:.1f}s / {a['seconds_max']:.1f}s",
              f"answered           {b['answered']}/{n} -> {a['answered']}/{n}"]
    return "\n".join(lines)


def format_stats(result: dict) -> str:
    if not result["requests"]:
        return "No requests match."
    n = result["requests"]
    lines = [
        f"requests            {n}",
        f"cost per request    mean ${result['cost_per_request_mean']:.6f}   p50 ${result['cost_per_request_p50']:.6f}   "
        f"p95 ${result['cost_per_request_p95']:.6f}   total ${result['cost_total']:.5f}",
        f"tokens per request  {result['tokens_per_request_mean']:.0f}   LLM calls per request "
        f"{result['llm_calls_per_request_mean']:.2f}",
        f"latency             p50 {result['seconds_p50']:.1f}s   p95 {result['seconds_p95']:.1f}s   "
        f"rate-limited {result['rate_limited_requests']}/{n}",
        f"outcomes            " + ", ".join(f"{k} {v}" for k, v in sorted(result["outcomes"].items())),
        "",
        f"{'step':<28} {'count':>6} {'per req':>8} {'time':>9} {'time %':>7} {'cost':>11} {'cost %':>7}",
    ]
    total_seconds = sum(s["seconds"] for s in result["by_step"].values()) or 1
    total_cost = sum(s["cost"] for s in result["by_step"].values()) or 1
    for name, s in result["by_step"].items():
        lines.append(f"{name:<28} {s['count']:>6.0f} {s['count'] / n:>8.2f} {s['seconds']:>8.1f}s "
                     f"{100 * s['seconds'] / total_seconds:>6.1f}% ${s['cost']:>10.6f} {100 * s['cost'] / total_cost:>6.1f}%")
    return "\n".join(lines)


def audit(traces: list[dict], allergen: str | None = None) -> list[dict]:
    """Every answer that calls a dish free from an allergen the ingredient database says it contains.

    Checks the *answer text* against the database, not against what the
    agent looked up -- so it also finds the case where the agent never
    looked the dish up at all. `looked_up` says whether it did.
    """
    from agents.allergen_claims import allergen_facts, free_from_claims, name_index
    from mcp_servers.ingredient_server import load_dishes

    dishes = load_dishes()
    index = name_index({d.dish: d.aliases for d in dishes})
    database = {d.dish: set(d.allergens) for d in dishes}
    findings = []
    for trace in traces:
        claims = [c for c in free_from_claims(trace.get("answer") or "", index)
                  if c.allergen in database.get(c.dish, set()) and (allergen is None or c.allergen == allergen)]
        if not claims:
            continue
        looked_up = allergen_facts(s.get("attrs", {}).get("result") for s in trace.get("spans", []) if s["kind"] == "tool")
        for claim in claims:
            findings.append({"request_id": trace["request_id"], "timestamp": trace["timestamp"],
                             "user_id": trace.get("user_id"), "dish": claim.dish, "allergen": claim.allergen,
                             "looked_up": claim.dish in looked_up, "text": claim.text})
    return sorted(findings, key=lambda f: f["timestamp"], reverse=True)


def recheck(traces: list[dict]) -> list[tuple[dict, list[str]]]:
    """Run the *current* output validation over logged answers and their logged tool results.

    Before shipping a new check, this shows what it would have done to real
    traffic: which bad answers it catches, and -- just as important -- which
    good answers it would wrongly block.
    """
    from agents.agent import AgentStep
    from agents.output_validation import validate_answer

    flagged = []
    for trace in traces:
        if trace.get("outcome") != "done" or not trace.get("answer"):
            continue
        steps = [AgentStep(tool=s["name"].split(":", 1)[1], args=s["attrs"].get("args"),
                           result=s["attrs"].get("result"), blocked=s["attrs"].get("blocked"))
                 for s in trace.get("spans", []) if s["kind"] == "tool"]
        issues = validate_answer(trace["answer"], steps, "")
        if issues:
            flagged.append((trace, issues))
    return flagged


def rejected_drafts_now_passing(traces: list[dict]) -> list[tuple[dict, list[str]]]:
    """Drafts a logged check rejected that the *current* check would accept -- with the issues it raised then.

    The other half of recheck(): a check that blocks good answers costs a
    retry at best and the whole answer at worst. Each draft is replayed with
    only the tool results that existed when it was checked.
    """
    from agents.agent import AgentStep
    from agents.output_validation import validate_answer

    cleared = []
    for trace in traces:
        spans = trace.get("spans", [])
        for check in spans:
            if check["kind"] != "check" or not check["attrs"].get("issues") or not check["attrs"].get("answer"):
                continue
            steps = [AgentStep(tool=s["name"].split(":", 1)[1], args=s["attrs"].get("args"),
                               result=s["attrs"].get("result"), blocked=s["attrs"].get("blocked"))
                     for s in spans if s["kind"] == "tool" and s["start_ms"] < check["start_ms"]]
            if not validate_answer(check["attrs"]["answer"], steps, ""):
                cleared.append((trace, check["attrs"]["issues"]))
    return cleared


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Search and inspect the request traces.")
    parser.add_argument("--log", type=Path, default=TRACE_PATH, help="Trace file (default data/traces.jsonl).")
    sub = parser.add_subparsers(dest="command", required=True)

    def add_filters(p: argparse.ArgumentParser) -> None:
        p.add_argument("--user")
        p.add_argument("--session")
        p.add_argument("--route")
        p.add_argument("--outcome")
        p.add_argument("--tool", help="Only requests that called this tool.")
        p.add_argument("--since", help="'3d', '12h', or an ISO date/time.")
        p.add_argument("--until", help="'1d', or an ISO date/time.")
        p.add_argument("--min-cost", type=float)
        p.add_argument("--min-seconds", type=float)

    find = sub.add_parser("find", help="List requests matching words and filters, newest first.")
    find.add_argument("words", nargs="*", help="Words that must all appear in the question or answer.")
    find.add_argument("--limit", type=int, default=20)
    add_filters(find)
    show = sub.add_parser("show", help="Show one request step by step.")
    show.add_argument("request_id")
    show.add_argument("--full", action="store_true", help="Don't shorten tool results.")
    show.add_argument("--json", action="store_true", help="Print the raw logged JSON.")
    st = sub.add_parser("stats", help="Cost and latency per request, and per step.")
    st.add_argument("--reprice", action="store_true",
                    help="Recompute cost from logged tokens at today's prices (compare runs logged before a price fix).")
    add_filters(st)
    pr = sub.add_parser("promote", help="Turn a logged failure into a live regression case.")
    pr.add_argument("request_id")
    pr.add_argument("reason", help="What went wrong, in a few words.")
    au = sub.add_parser("audit", help="Answers claiming a dish is free from an allergen it contains.")
    au.add_argument("--allergen", help="Only this allergen, e.g. dairy.")
    add_filters(au)
    cp = sub.add_parser("compare", help="Same questions, two runs: cost, calls, outcomes.")
    cp.add_argument("before_session", help="Session id of the first run ('-' for requests with no session).")
    cp.add_argument("after_session", help="Session id of the second run.")
    rc = sub.add_parser("recheck", help="Which past answers would the current output checks reject?")
    add_filters(rc)
    args = parser.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    traces = read_traces(args.log)
    if args.command == "show":
        matches = [t for t in traces if t["request_id"].startswith(args.request_id)]
        if len(matches) != 1:
            print(f"{len(matches)} requests match {args.request_id!r}.", file=sys.stderr)
            return 1
        print(json.dumps(matches[0], indent=2, ensure_ascii=False) if args.json else format_trace(matches[0], full=args.full))
        return 0

    if args.command == "compare":
        def session(name: str) -> list[dict]:
            wanted = None if name == "-" else name
            return [t for t in traces if t.get("session_id") == wanted]
        print(format_compare(compare(session(args.before_session), session(args.after_session))))
        return 0
    if args.command == "promote":
        from observability.regressions import add_case

        matches = [t for t in traces if t["request_id"].startswith(args.request_id)]
        if len(matches) != 1:
            print(f"{len(matches)} requests match {args.request_id!r}.", file=sys.stderr)
            return 1
        try:
            case = add_case(matches[0], args.reason)
        except ValueError as error:
            print(error, file=sys.stderr)
            return 1
        print(f"added regression case from {case['source_request_id']}: {case['question']}")
        return 0

    filters = dict(
        user=args.user, session=args.session, route=args.route, outcome=args.outcome, tool=args.tool,
        since=parse_when(args.since) if args.since else None, until=parse_when(args.until) if args.until else None,
        min_cost=args.min_cost, min_seconds=args.min_seconds,
    )
    if args.command == "audit":
        selected = filter_traces(traces, **filters)
        findings = audit(selected, args.allergen)
        for f in findings:
            when = _when(f).strftime("%Y-%m-%d %H:%M")
            print(f"{f['request_id']}  {when}  {(f['user_id'] or '-'):<8} says {f['dish']!r} is {f['allergen']}-free "
                  f"-- the database lists {f['allergen']}  (looked up in this request: {'yes' if f['looked_up'] else 'NO'})")
            print(f"    \"{_one_line(f['text'], 160)}\"")
        print(f"\n{len(findings)} false allergen claim(s) in {len(selected)} requests checked")
        return 0
    if args.command == "recheck":
        selected = filter_traces(traces, **filters)
        flagged = recheck(selected)
        for trace, issues in flagged:
            print(format_row(trace))
            for issue in issues:
                print(f"    {issue}")
        checked = sum(1 for t in selected if t.get("outcome") == "done")
        print(f"\n{len(flagged)} of {checked} answered requests would now be rejected")
        cleared = rejected_drafts_now_passing(selected)
        for trace, issues in cleared:
            print(format_row(trace))
            print(f"    rejected then for: {'; '.join(issues)} -- the current check accepts it")
        print(f"{len(cleared)} draft(s) rejected at the time would now pass")
        return 0
    if args.command == "find":
        hits = filter_traces(traces, words=args.words, **filters)
        for trace in hits[: args.limit]:
            print(format_row(trace))
        print(f"\n{len(hits)} of {len(traces)} requests match" + (f" (showing {args.limit})" if len(hits) > args.limit else ""))
        return 0
    selected = filter_traces(traces, **filters)
    print(format_stats(stats([reprice(t) for t in selected] if args.reprice else selected)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
