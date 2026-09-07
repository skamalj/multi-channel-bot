"""Agent specifications. An agent IS its tag set.

Binding happens once, at construction. Nothing about a conversation changes
which tools an agent has - which is what makes the capability matrix in
docs/architecture.md printable.
"""
from __future__ import annotations

from pydantic import BaseModel, Field

Persona = str  # "customer" | "agent"
Lob = str      # "health" | "motor" | "property" | "uc"


class AgentSpec(BaseModel):
    bot_id: str
    name: str
    persona: Persona
    lob: Lob

    # The whole tool filter. Build time, static.
    tool_tags: dict[str, str] = Field(default_factory=dict)

    # Retrieval scope for this configuration.
    corpus_scope: list[str] = Field(default_factory=lambda: ["public"])

    system_prompt: str = ""
    greeting: str = ""

    @property
    def key(self) -> str:
        return f"{self.persona}#{self.lob}"
