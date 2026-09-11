"""ONE MCP server hosting every tool (AG-1).

One deployment, one Cloud Map entry, one CI pipeline. Agents do not each get a
server; they filter THIS server's tool list by tag at build time. In-process
they bind the registry's `@tool` objects directly, which is a local transport
over the same manifest - `describe()` is what both paths read.

Run it as a real MCP stdio server:

    uv run python -m app.mcpserver.server --serve

Print the manifest instead:

    uv run python -m app.mcpserver.server
"""
from __future__ import annotations

import json
import sys

from langchain_core.utils.function_calling import convert_to_openai_tool

from app.mcpserver.registry import all_tools, get_tool


def describe() -> list[dict]:
    """The manifest: the model-visible schema and the behaviour policy from
    `extras`, side by side. The schema is what the model receives; the policy
    (effect/authority/auth/consent/subject/tags) is read server-side by the
    authorization hook and never sent to the model."""
    out = []
    for t in sorted(all_tools(), key=lambda s: s.name):
        fn = convert_to_openai_tool(t)["function"]
        extras = t.extras or {}
        out.append({
            "name": t.name,
            "description": fn.get("description", ""),
            "input_schema": fn.get("parameters", {}),
            "tags": extras.get("tags", {}),
            "effect": extras.get("effect", "read"),
            "authority": extras.get("authority", "none"),
            "pii": extras.get("pii", False),
            "auth": extras.get("auth", "anonymous"),
            "subject": extras.get("subject", "none"),
            "consent_purpose": extras.get("consent_purpose"),
            "hitl_required": extras.get("hitl_required", False),
        })
    return out


def serve() -> None:                                         # pragma: no cover
    """Expose the same registry over MCP stdio.

    Note what is NOT here: an authorization decision. Over a wire the caller
    context arrives with the request, and the authorization hook runs against
    it in the calling layer. A tool server that trusts its transport is a tool
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
        tool = get_tool(name)
        if tool is None:
            return [TextContent(type="text",
                                text=json.dumps({"error": "unknown_tool",
                                                 "name": name}))]
        try:
            result = tool.invoke(arguments)
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
