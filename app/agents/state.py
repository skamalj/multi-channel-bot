"""Per-bot conversation state - the checkpointed channel set (ME-2).

Thread id is `user#lob`. Everything in this TypedDict is persisted by the
checkpointer between turns; everything per-turn travels in the config and is
deliberately NOT here.

That distinction is the one to get right. `ctx` carries the idempotency key,
the consent snapshot and the caller's identity, all of which are true for
exactly one turn. Checkpointed, they would come back on the next turn and the
double-write protection would invert into a double-write cause.
"""
from __future__ import annotations

from typing import Annotated, Any, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


class BotState(TypedDict, total=False):
    # The reducer is real: nodes return {"messages": [msg]} and LangGraph
    # merges. Returning a message with an existing id REPLACES it, which is
    # how the citation guardrail rewrites an answer in place.
    messages: Annotated[list[AnyMessage], add_messages]

    # Reply addressing - carried from the ingest event so the respond node
    # never needs a lookup. An error can always be delivered.
    user_id: str
    channel: str
    channel_identity: str | None
    last_message_id: str

    # Configuration chosen by the resolver.
    bot_id: str
    persona: str
    lob: str
    corpus_scope: list[str]

    # Shared profile passed in per turn - facts ABOUT a line of business.
    shared: dict[str, Any]

    # Journey state captured in THIS line of business, and persisted.
    # `documents` holds METADATA only - the bytes are in object storage, and
    # a document in the message list would push a checkpoint past the 1 MB a
    # DynamoDB query returns all by itself.
    documents: list[dict]
    slots: dict[str, Any]
    quotes: dict[str, Any]
    # AG-6: waiting is a field that is still empty, not a parked node.
    pending_confirmation: dict[str, Any] | None
    # Facts this journey established, so a later turn can check what the
    # model says about them. Bounded in the respond node.
    known_facts: list[str]

    # --- per-turn scratch, reset by the entry node ----------------------
    # These are in the checkpoint because everything in a graph's state is,
    # but they carry no meaning across turns and are cleared on the way in.
    rounds: int
    retrieved: list[dict]
    tool_facts: list[str]
    tools_called: list[str]
    used_core_tool: bool
    looked_up: bool
    parked: bool
    last_tools: list[str]
    # Set by the entry node when Bedrock Guardrails rejected the inbound
    # message. Per-turn scratch like the rest of this block.
    inbound_blocked: bool
    last_citations: list[str]
