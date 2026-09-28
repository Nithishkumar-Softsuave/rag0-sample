"""Shared launch code for this project's MCP servers: pick a transport, and
lock down the remote one.

Two transports, same server code:

- stdio: the host launches the server as a subprocess and talks over its
  stdin/stdout. Nothing listens on the network, so access control is "who
  can start the process" -- the OS handles it. This is what the local agent
  uses (mcp_config.json).
- streamable-http: the server listens on a URL so *someone else's* agent --
  a teammate's host, Claude Code, Claude Desktop -- can connect over the
  network. Anything reachable over a network needs auth, so HTTP mode
  refuses to start without a bearer token (fail closed), and every request
  must carry `Authorization: Bearer <token>`.

The token check is the MCP spec's resource-server role: the server only
*verifies* tokens (`TokenVerifier`); it never issues them. Here the verifier
compares against static tokens from an environment variable, which is the
honest size for a PoC. In production you would swap `StaticTokenVerifier`
for one that validates JWTs from your identity provider (checking issuer,
audience and expiry) -- the rest of this file stays the same.
"""
from __future__ import annotations

import argparse
import hmac
import logging
import os
from collections.abc import Callable

from mcp.server.auth.provider import AccessToken
from mcp.server.auth.settings import AuthSettings
from mcp.server.mcpserver import MCPServer
from mcp.server.transport_security import TransportSecuritySettings

logger = logging.getLogger(__name__)

# Server-side logs go to stderr (stdout is the stdio transport's wire), so
# the default is quiet enough not to drown the host's own terminal output.
LOG_LEVEL = os.getenv("MCP_SERVER_LOG_LEVEL", "WARNING").upper()


class StaticTokenVerifier:
    """Accept only bearer tokens listed in one environment variable.

    `tokens` maps token -> client id, so the server can tell callers apart in
    its logs (and revoke one teammate without affecting another). Comparison
    is constant-time so a wrong token leaks nothing about the right one.
    """

    def __init__(self, tokens: dict[str, str], scopes: list[str], resource: str) -> None:
        self._tokens = tokens
        self._scopes = scopes
        self._resource = resource

    async def verify_token(self, token: str) -> AccessToken | None:
        for known, client_id in self._tokens.items():
            if hmac.compare_digest(token.encode(), known.encode()):
                # These tokens are minted for this one server, so their
                # audience is this server's URL -- which lets the SDK's
                # validate_token_resource check refuse tokens meant for
                # any other resource.
                return AccessToken(token=token, client_id=client_id, scopes=list(self._scopes), resource=self._resource)
        return None


def parse_tokens(raw: str) -> dict[str, str]:
    """Parse "alice:tok1,bob:tok2" (or a bare "tok1") into {token: client_id}."""
    tokens: dict[str, str] = {}
    for index, entry in enumerate(part.strip() for part in raw.split(",")):
        if not entry:
            continue
        client_id, _, token = entry.rpartition(":")
        tokens[token] = client_id or f"client-{index + 1}"
    return tokens


def http_auth(token_env: str, scope: str, public_url: str) -> tuple[StaticTokenVerifier, AuthSettings]:
    """Build the verifier + auth settings for HTTP mode, or refuse to start.

    `issuer_url` is only advertised in the server's protected-resource
    metadata (where a real client would go to obtain a token); nothing
    here is an authorization server.
    """
    tokens = parse_tokens(os.getenv(token_env, ""))
    if not tokens:
        raise SystemExit(
            f"Refusing to serve over HTTP without auth: set {token_env} to one or more bearer tokens, "
            f'e.g. {token_env}="teammate:{os.urandom(16).hex()}". Use --transport stdio for local-only use.'
        )
    verifier = StaticTokenVerifier(tokens, scopes=[scope], resource=public_url)
    settings = AuthSettings(
        issuer_url=public_url, resource_server_url=public_url, required_scopes=[scope], validate_token_resource=True,
    )
    return verifier, settings


def run(
    build_server: Callable[..., MCPServer],
    *,
    description: str,
    default_port: int,
    token_env: str,
    scope: str,
) -> None:
    """Command-line entry point shared by every server in this package."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--transport", choices=["stdio", "http"], default="stdio")
    parser.add_argument("--host", default="127.0.0.1", help="HTTP only. Use 0.0.0.0 to accept teammates on your LAN.")
    parser.add_argument("--port", type=int, default=default_port, help="HTTP only.")
    parser.add_argument(
        "--allowed-host", action="append", default=[],
        help="HTTP only. Extra Host header value to accept (e.g. 192.168.1.20:8765). "
             "DNS-rebinding protection rejects anything not listed.",
    )
    args = parser.parse_args()

    if args.transport == "stdio":
        build_server().run("stdio")
        return

    # The URL teammates will use: the first --allowed-host (e.g. your LAN
    # address) if given, else the bind address. It is advertised in the
    # protected-resource metadata and is the audience tokens are bound to.
    if args.allowed_host:
        public_url = f"http://{args.allowed_host[0]}/mcp"
    else:
        public_host = args.host if args.host not in {"0.0.0.0", "::"} else "localhost"
        public_url = f"http://{public_host}:{args.port}/mcp"
    verifier, auth = http_auth(token_env, scope, public_url)
    allowed_hosts = [f"127.0.0.1:{args.port}", f"localhost:{args.port}", *args.allowed_host]
    security = TransportSecuritySettings(enable_dns_rebinding_protection=True, allowed_hosts=allowed_hosts)

    server = build_server(token_verifier=verifier, auth=auth)
    print(f"[{server.name}] serving MCP over streamable HTTP at {public_url} (bearer token required)", flush=True)
    server.run("streamable-http", host=args.host, port=args.port, transport_security=security)
