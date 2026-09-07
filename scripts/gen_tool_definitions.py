"""Generate the AgentCore Gateway ToolDefinition list from the tool registry.

The registry already knows every tool's real argument types - `json_schema()`
resolves them with `get_type_hints`, so `int | None` stays an integer instead
of collapsing into a string. Writing these 27 definitions by hand would mean
maintaining a second, worse copy of that knowledge, and the copy would drift.

Run at build time; the output goes to S3 and the Gateway target points at it.

    uv run python scripts/gen_tool_definitions.py --out tool-definitions.json
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from app.mcpserver.registry import (  # noqa: E402
    ensure_loaded, json_schema, list_tools,
)

# Gateway tool names are stricter than Python identifiers.
NAME_MAX = 64


def _description(spec) -> str:
    """What the model reads to decide whether to call this tool.

    The effect and authority are stated in the description because they
    change how a model should treat the tool: a `write` with `confirm` is not
    something to try speculatively.
    """
    base = (spec.fn.__doc__ or spec.name).strip().split("\n\n")[0]
    base = " ".join(base.split())
    bits = [base]
    if getattr(spec, "effect", None) and spec.effect != "read":
        bits.append(f"Effect: {spec.effect}.")
    if getattr(spec, "confirm", False):
        bits.append("Requires explicit customer confirmation before calling.")
    text = " ".join(bits)
    return text[:1000]


def build() -> list[dict]:
    ensure_loaded()
    out: list[dict] = []
    for spec in sorted(list_tools(), key=lambda s: s.name):
        name = spec.name[:NAME_MAX]
        schema = json_schema(spec)
        out.append({
            "name": name,
            "description": _description(spec),
            "inputSchema": schema,
            # The Gateway wants an output schema. Ours is uniform: every tool
            # returns a JSON object, and what is IN it is the tool's business
            # and is checked downstream by the citation guardrail, not here.
            "outputSchema": {
                "type": "object",
                "description": "Tool result. `ok` false means it did not run.",
            },
        })
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="tool-definitions.json")
    args = ap.parse_args()

    defs = build()
    if not defs:
        # An empty list would deploy a Gateway that silently exposes nothing.
        print("no tools found - is the registry importing its modules?",
              file=sys.stderr)
        return 1

    path = pathlib.Path(args.out)
    path.write_text(json.dumps(defs, indent=2), encoding="utf-8")
    print(f"{len(defs)} tool definitions -> {path} "
          f"({path.stat().st_size} bytes)")
    for d in defs:
        req = d["inputSchema"].get("required") or []
        print(f"  {d['name']:<28} {len(d['inputSchema'].get('properties', {})):>2} args"
              f"  {len(req)} required")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
