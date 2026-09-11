"""Tool registry: collect the framework @tool objects and bind them by tag.

Two questions, kept apart:

| May this bot ever hold this tool?            | build time | here, by tag        |
| May this caller make this call, now?         | per call   | app/agents/controls |

The tools are plain LangChain `@tool` objects (app/mcpserver/tools/*). This
module imports the modules that define them, harvests the `BaseTool`
instances, and filters by the `tags` they carry in `extras`. There is no
ToolSpec and no parallel schema any more: the tool IS its definition - name,
description, typed arguments - and its policy (effect, authority, auth,
consent, subject, tags) rides in `extras`, read at runtime by the
authorization hook and the audit record, never sent to the model.

A model can still NAME a tool it was never bound - that is why the hook
re-checks authority per call. Binding is a design boundary; the hook is the
breach boundary.
"""
from __future__ import annotations

from langchain_core.tools import BaseTool

_TOOLS: dict[str, BaseTool] = {}
_LOADED = False


def ensure_loaded() -> None:
    """Importing a tool module is what defines its tools; harvesting them here
    means the manifest is complete no matter who asks first."""
    global _LOADED
    if _LOADED:
        return
    _LOADED = True
    from app.mcpserver.tools import (issuance, knowledge,  # noqa: F401
                                     policy, product)
    for module in (product, knowledge, policy, issuance):
        for obj in vars(module).values():
            if isinstance(obj, BaseTool):
                _TOOLS[obj.name] = obj


def all_tools() -> list[BaseTool]:
    ensure_loaded()
    return list(_TOOLS.values())


def get_tool(name: str) -> BaseTool | None:
    ensure_loaded()
    return _TOOLS.get(name)


def tools_for(want: dict[str, str] | None = None) -> list[BaseTool]:
    """The tools a bot binds: every one whose tags satisfy the bot's tag set."""
    ensure_loaded()
    if not want:
        return list(_TOOLS.values())
    return [t for t in _TOOLS.values()
            if _matches((t.extras or {}).get("tags", {}), want)]


def _matches(spec_tags: dict[str, str], want: dict[str, str]) -> bool:
    """A tool matches when every requested tag is satisfied. "*" means any;
    a "a|b" tag value is satisfied by either."""
    for key, wanted in want.items():
        have = spec_tags.get(key)
        if have is None:
            return False
        if have == "*" or wanted == "*":
            continue
        if wanted not in {v.strip() for v in have.split("|")}:
            return False
    return True
