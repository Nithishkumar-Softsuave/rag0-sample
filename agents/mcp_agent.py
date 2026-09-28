"""Run the Week 7/8 agent with tools discovered over MCP instead of imported.

    python -m agents.mcp_agent "Which is cheaper, filter coffee or badam milk?"
    python -m agents.mcp_agent --list                        # just show what was discovered
    python -m agents.mcp_agent --server menu "..."           # connect to a subset of servers
    python -m agents.mcp_agent --config other.json "..."

The loop is agents/agent.py's run_agent(), unchanged, with the HARDENED
Week 8 policy (tool guards, untrusted-output preamble, output validation).
The only difference from agents/race.py's agent is where tool_schemas,
tool_functions and the system prompt come from: McpHost (agents/mcp_host.py).
"""
from __future__ import annotations

import argparse
import json
import sys

from agents.agent import run_agent
from agents.mcp_host import DEFAULT_CONFIG, McpHost, load_config


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the menu agent over MCP-discovered tools.")
    parser.add_argument("question", nargs="?", help="Question for the agent.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="MCP config file (mcpServers format).")
    parser.add_argument("--server", action="append", help="Only connect to this server (repeatable).")
    parser.add_argument("--list", action="store_true", help="List discovered tools and exit.")
    args = parser.parse_args(argv)
    # Model answers can contain non-cp1252 characters (e.g. narrow spaces);
    # the Windows console default would crash on print.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if not args.list and not args.question:
        parser.error("give a question, or --list")

    configs = load_config(args.config)
    if args.server:
        unknown = set(args.server) - {config.name for config in configs}
        if unknown:
            parser.error(f"not in {args.config}: {sorted(unknown)}")
        configs = [config for config in configs if config.name in args.server]

    with McpHost(configs) as host:
        print("Discovered over MCP:")
        print(host.describe())
        if args.list:
            for name, discovered in host.tools.items():
                print(f"\n{name}  [{discovered.server}]  {discovered.fingerprint}\n  {discovered.tool.description}")
            return 0
        if not host.tools:
            print("No tools available -- nothing to run the agent with.", file=sys.stderr)
            return 1

        result = run_agent(
            args.question,
            system_prompt=host.system_prompt,
            tool_schemas=host.tool_schemas,
            tool_functions=host.tool_functions,
        )

    print("\nTrajectory:")
    for index, step in enumerate(step for step in result.steps if step.tool):
        status = f"BLOCKED ({step.blocked})" if step.blocked else "ok"
        print(f"  {index + 1}. {step.tool}({json.dumps(step.args)}) -> {status}")
        print(f"     {json.dumps(step.result)[:300]}")
    print(f"\nAnswer ({result.stopped_reason}, {result.tool_calls} tool calls, "
          f"{result.elapsed_seconds:.1f}s, ${result.cost_usd:.4f}):\n{result.answer}")
    if result.validation_issues:
        print(f"Validation issues: {result.validation_issues}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
