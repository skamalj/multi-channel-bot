"""The per-invocation request context.

Passed as the agent's `context` at invoke time (`create_agent(...,
context_schema=RequestContext)`), read inside tools via `runtime.context` and
inside hooks via `request.runtime.context`. It is NEVER part of a
model-visible schema: an entitlement the model can see is an entitlement it
can argue its way around, so identity, consent and corpus scope arrive here,
out of band, and the model only ever names business arguments.

This is deliberately not in `BotState`. State is checkpointed and comes back
next turn; identity and consent are true for exactly one turn and must be
supplied fresh each time - a stale customer id in a checkpoint is the kind of
thing that reads someone else's book.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class RequestContext:
    # Who is calling. `user_id` is the channel identity; `customer_id` is the
    # core customer once resolved; `producer_id` is set only for an agent.
    user_id: str | None = None
    customer_id: str | None = None
    producer_id: str | None = None

    # Which bot this is, which compartment it may touch.
    persona: str = "customer"
    lob: str | None = None
    corpus_scope: list[str] = field(default_factory=lambda: ["public"])

    # Entitlement level reached this turn, and the consent on record. Both are
    # read by the authorization hook; neither is ever taken from the model.
    authenticated: bool = False
    consent: dict[str, bool] = field(default_factory=dict)

    # The per-turn trace sink. Per-invocation like the rest of this object,
    # never checkpointed; hooks reach it via `runtime.context.trace`.
    trace: Any = None
