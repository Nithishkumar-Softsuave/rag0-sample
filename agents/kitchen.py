"""Week 11: the kitchen assistant as a production entry point -- every request is traced.

The same agents as Weeks 9-10 (single agent over MCP tools by default, the
squad on request), wrapped in observability/tracing.py's request(): each
question gets a request id, and its full trace -- question, answer, every
LLM call, tool call and check, with time and cost per step -- is appended to
data/traces.jsonl. The request id is printed with the answer: it is what a
customer (or support) quotes to find this exact answer later.

    python -m agents.kitchen "Is Masala Dosa dairy-free?"
    python -m agents.kitchen --route squad --user u_042 "..."
    python -m agents.kitchen --batch traffic.txt --pause 2     # one question per line,
                                                             # optional "user_id<TAB>question"

Find it again with observability/logs.py:

    python -m observability.logs find dairy swap
    python -m observability.logs show req_1a2b3c4d5e6f
"""
from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from agents.agent import AgentResult, run_agent
from agents.mcp_host import DEFAULT_CONFIG, McpHost, load_config
from agents.squad import SquadResult, run_squad
from observability.tracing import Trace, prompt_version, request

ROUTES = ("single", "squad")


@dataclass
class Reply:
    request_id: str
    answer: str
    outcome: str
    trace: Trace
    result: AgentResult | SquadResult


def ask(question: str, host: McpHost, *, route: str = "single", user_id: str | None = None,
        session_id: str | None = None, trace_path: Path | None = None) -> Reply:
    """Answer one customer question and log its trace. The only way the app answers anything."""
    if route not in ROUTES:
        raise ValueError(f"route must be one of {ROUTES}")
    tools = {name: tool.fingerprint for name, tool in host.tools.items()}
    with request(question, route=route, user_id=user_id, session_id=session_id, path=trace_path,
                 tools=tools, host_prompt_version=prompt_version(host.system_prompt),
                 mcp_failures=dict(host.failures)) as trace:
        if route == "single":
            result: AgentResult | SquadResult = run_agent(
                question, system_prompt=host.system_prompt,
                tool_schemas=host.tool_schemas, tool_functions=host.tool_functions,
            )
        else:
            result = run_squad(question, host)
            trace.set(plan=result.plan)
        trace.answer = result.answer
        trace.outcome = result.stopped_reason
        trace.set(tool_calls=[step.tool for step in result.steps if step.tool],
                  blocked_tool_calls=sum(1 for step in result.steps if step.blocked),
                  validation_issues=result.validation_issues)
    return Reply(request_id=trace.request_id, answer=trace.answer, outcome=trace.outcome, trace=trace, result=result)


def _read_batch(path: Path) -> list[tuple[str | None, str]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        user, _, question = line.partition("\t")
        rows.append((user, question) if question else (None, user))
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Ask the kitchen assistant; every request is traced.")
    parser.add_argument("question", nargs="?")
    parser.add_argument("--route", choices=ROUTES, default="single")
    parser.add_argument("--user", help="Opaque user id to log with the request.")
    parser.add_argument("--session", help="Session id to log with the request.")
    parser.add_argument("--batch", type=Path, help="File of questions, one per line ('user<TAB>question' allowed).")
    parser.add_argument("--pause", type=float, default=0.0, help="Seconds between batch questions (rate limits).")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="MCP config file.")
    args = parser.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if not args.question and not args.batch:
        parser.error("give a question or --batch")

    rows = _read_batch(args.batch) if args.batch else [(args.user, args.question)]
    with McpHost(load_config(args.config)) as host:
        for index, (user, question) in enumerate(rows):
            if index and args.pause:
                time.sleep(args.pause)
            try:
                reply = ask(question, host, route=args.route, user_id=user or args.user, session_id=args.session)
            except Exception as exc:  # logged as outcome "error" by request(); keep the batch going
                print(f"[error] {question[:60]!r}: {type(exc).__name__}: {exc}", file=sys.stderr)
                continue
            t = reply.trace
            print(f"\n{reply.request_id}  {reply.outcome}  {t.duration_ms / 1000:.1f}s  "
                  f"{t.prompt_tokens + t.completion_tokens} tokens  ${t.cost_usd:.5f}")
            print(f"Q: {question}\nA: {reply.answer}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
