"""Speak MCP by hand, once, so it stops being a mystery.

    python -m scripts.mcp_raw_messages                  # menu server
    python -m scripts.mcp_raw_messages --server ingredient_server

No MCP SDK on this side: it starts a server as a subprocess and writes
JSON-RPC 2.0 messages to its stdin, one per line, printing everything that
goes each way. That is all the stdio transport is. The exchange:

  1. initialize                 client -> server   "I speak protocol X, here is who I am"
     (result)                   server -> client   "I speak X too; I have tools/resources/prompts; instructions: ..."
  2. notifications/initialized  client -> server   handshake done (a notification: no id, no reply)
  3. tools/list                 client -> server   discovery -- the whole point of MCP
  4. tools/call                 client -> server   run one tool with arguments
  5. tools/call (bad args)      client -> server   a recoverable error comes back as a normal
                                                   result with isError: true, not a crash

Uses the 2025-11-25 protocol revision, the newest one with the initialize
handshake. (2026-07-28 replaces the handshake with a stateless per-request
envelope; the SDK negotiates it automatically, which is why
scripts/mcp_inspect.py reports it. The discovery and call messages are the
same idea either way.)
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_VERSION = "2025-11-25"

EXAMPLE_CALLS = {
    "menu_server": [("search_menus", {"item": "Filter Coffee"}), ("convert_currency", {"amount": 34, "from_currency": "USD", "to_currency": "INR"})],
    "ingredient_server": [("lookup_ingredients", {"dish": "Ven Pongal"}), ("lookup_ingredients", {"dish": "Pizza"})],
}


def show(direction: str, message: dict) -> None:
    print(f"\n{direction}\n{json.dumps(message, indent=2, ensure_ascii=False)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--server", choices=sorted(EXAMPLE_CALLS), default="menu_server")
    args = parser.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    process = subprocess.Popen(
        [sys.executable, "-m", f"mcp_servers.{args.server}"],
        cwd=PROJECT_ROOT, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, encoding="utf-8",
    )
    assert process.stdin and process.stdout
    next_id = 0

    def send(message: dict) -> None:
        show(">>> client -> server", message)
        process.stdin.write(json.dumps(message) + "\n")
        process.stdin.flush()

    def request(method: str, params: dict) -> dict:
        nonlocal next_id
        next_id += 1
        send({"jsonrpc": "2.0", "id": next_id, "method": method, "params": params})
        while True:  # skip any server notifications until our response arrives
            message = json.loads(process.stdout.readline())
            show("<<< server -> client", message)
            if message.get("id") == next_id:
                return message

    try:
        request("initialize", {
            "protocolVersion": PROTOCOL_VERSION,
            "capabilities": {},
            "clientInfo": {"name": "raw-json-rpc-demo", "version": "0.1"},
        })
        send({"jsonrpc": "2.0", "method": "notifications/initialized"})
        request("tools/list", {})
        for name, arguments in EXAMPLE_CALLS[args.server]:
            request("tools/call", {"name": name, "arguments": arguments})
    finally:
        process.stdin.close()  # closing stdin is how a stdio client says goodbye
        process.wait(timeout=10)


if __name__ == "__main__":
    main()
