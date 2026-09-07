"""The resolver session - the shape checkpointed on the `user` thread (ME-1).

Holds the route ledger, the shared profile and auth state. Long-lived and
low-sensitivity: it may contain facts ABOUT a line of business, never facts
FROM one.

There is no store class here any more. LangGraph's checkpointer persists this
model on the resolver thread, so this file describes the shape and nothing
else - which is the point of having a checkpointer at all.
"""
from __future__ import annotations

from pydantic import BaseModel, Field

from app.resolver.ledger import RouteLedger


class SharedProfile(BaseModel):
    """The ONLY thing that crosses a bot-session boundary."""

    display_name: str | None = None
    city: str | None = None
    language: str = "en"
    channel: str | None = None
    authenticated: bool = False
    consent: dict[str, bool] = Field(default_factory=dict)
    # Facts ABOUT a line of business - never facts FROM one.
    holdings: dict[str, bool] = Field(default_factory=dict)


class ResolverSession(BaseModel):
    user_id: str
    persona: str | None = None
    active_lob: str | None = None
    ledger: RouteLedger = Field(default_factory=RouteLedger)
    profile: SharedProfile = Field(default_factory=SharedProfile)
    turn: int = 0

    # RS-9. A step-up is time-boxed; a persona switch into `agent` needs one
    # that is still current, and the switch itself carries no journey state.
    persona_stepup_at: float | None = None
    persona_switched: bool = False
