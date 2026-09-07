"""ONE MCP server hosting every tool (AG-1).

One deployment, one Cloud Map entry, one CI pipeline, one middleware for
auth, idempotency, rate limiting and audit. Six servers is six connections
and six handshakes per Lambda cold start, paid on every conversation.

Agents do not each get a server; they filter THIS server's tool list by tag
at build time. In-process they call the registry directly, which is a local
transport over the same manifest - `describe()` is what both paths read.

Run it as a real MCP stdio server (for MCP Inspector, or for an agent
runtime that speaks MCP over a wire):

    uv run python -m app.mcpserver.server --serve

Print the manifest instead:

    uv run python -m app.mcpserver.server
"""
from __future__ import annotations

import json
import sys

# Importing the tool modules is what registers them.
from app.mcpserver.tools import issuance, knowledge, policy, product  # noqa: F401
from app.mcpserver.registry import get_tool, json_schema, list_tools


def describe() -> list[dict]:
    """The manifest. Selection tags and behaviour metadata, side by side."""
    return [
        {
            "name": t.name,
            "description": t.description,
            "signature": t.signature,
            "input_schema": json_schema(t),
            "tags": t.tags,
            "effect": t.effect,
            "authority": t.authority,
            "pii": t.pii,
            "auth": t.auth,
            "subject": t.subject,
            "consent_purpose": t.consent_purpose,
            "confirm": t.confirm,
            "idempotent": t.idempotent,
        }
        for t in sorted(list_tools(), key=lambda s: s.name)
    ]


def serve() -> None:                                         # pragma: no cover
    """Expose the same registry over MCP stdio.

    Note what is NOT here: an authorization decision. Over a wire the caller
    context arrives with the request, and `authorize()` runs against it in
    the calling layer. A tool server that trusts its transport is a tool
    server with no control.
    """
    import anyio
    from mcp.server import Server
    from mcp.server.stdio import stdio_server
    from mcp.types import TextContent, Tool

    server = Server("protec-mcb")

    @server.list_tools()
    async def _list() -> list[Tool]:
        return [Tool(name=t["name"], description=t["description"],
                     inputSchema=t["input_schema"]) for t in describe()]

    @server.call_tool()
    async def _call(name: str, arguments: dict) -> list[TextContent]:
        spec = get_tool(name)
        if spec is None:
            return [TextContent(type="text",
                                text=json.dumps({"error": "unknown_tool",
                                                 "name": name}))]
        try:
            result = spec.fn(**arguments)
        except Exception as exc:                             # noqa: BLE001
            result = {"error": type(exc).__name__, "detail": str(exc)}
        return [TextContent(type="text",
                            text=json.dumps(result, default=str)[:16000])]

    async def _run():
        async with stdio_server() as (read, write):
            await server.run(read, write,
                             server.create_initialization_options())

    anyio.run(_run)


if __name__ == "__main__":                                   # pragma: no cover
    if "--serve" in sys.argv:
        serve()
    else:
        print(json.dumps(describe(), indent=2))
