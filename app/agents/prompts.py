"""Prompts, loaded from `prompts/` rather than concatenated in Python.

Every instruction the bot follows is a file somebody can read, diff and
review without reading the graph. That is the point: the workflow was
previously spread across string concatenation in `registry.py`, more strings
in `_system_prompt()`, and branches in the graph that injected sentences into
the conversation on the model's behalf. Nothing described the whole journey,
which is how a quote ended up asking for confirmation three times with no
single place that would have shown it.

A configuration's prompt is `common.md` plus its own file. Nothing else is
appended at call time except the two facts that come from the binding rather
than from the author - the line of business and the corpus scope - because
those are decided per turn and must match what the tools were actually
bound with.

The files are read once and cached. They ship in the container, so a change
to a prompt is a deploy, which is what `prompt_version` in the trace is for.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

# app/agents/prompts.py -> repository root
PROMPT_DIR = Path(__file__).resolve().parents[2] / "prompts"

COMMON = "common.md"


class PromptMissing(RuntimeError):
    """A configuration names a prompt file that is not there.

    Loud, and at build time. A bot silently running with half its
    instructions is worse than a bot that will not start: the missing half is
    invariably the part that says what not to do.
    """


@lru_cache(maxsize=None)
def _read(name: str) -> str:
    path = PROMPT_DIR / name
    if not path.is_file():
        raise PromptMissing(f"{path} does not exist")
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise PromptMissing(f"{path} is empty")
    return text


def load(name: str) -> str:
    """The common instructions plus one configuration's own."""
    return f"{_read(COMMON)}\n\n{_read(name)}"


def available() -> list[str]:
    return sorted(p.name for p in PROMPT_DIR.glob("*.md"))
