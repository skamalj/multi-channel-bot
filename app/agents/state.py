"""Per-bot conversation state.

The framework's `AgentState` already owns `messages` (with the `add_messages`
reducer). We extend it only with the few journey fields that must be
checkpointed. Everything per-turn - identity, consent, corpus scope, the
trace - travels as `RequestContext` (the agent's `context_schema`) and is
NEVER here: checkpointed, it would come back on the next turn, and a stale
identity is how a bot ends up reading someone else's book.

There is no per-turn scratch here any more (`rounds`, `retrieved`,
`tool_facts`, `tools_called`, `inbound_blocked`, ...). The react loop keeps
tool calls and their results in `messages`; the grounding guardrail reads the
retrieval passages back out of those `ToolMessage`s rather than from a
parallel copy that had to be cleared each turn.
"""
from __future__ import annotations

from typing_extensions import NotRequired

from langchain.agents.middleware import AgentState


class BotState(AgentState):
    # Journey state, persisted between turns.
    documents: NotRequired[list[dict]]      # METADATA only - bytes go to S3
    slots: NotRequired[dict]
    quotes: NotRequired[dict]
    # The cross-compartment profile, passed in per turn by the orchestrator.
    shared: NotRequired[dict]
    # Written by the grounding guardrail for the orchestrator's audit and the
    # interactions ledger - what the turn actually did and cited.
    last_tools: NotRequired[list[str]]
    last_citations: NotRequired[list[str]]
