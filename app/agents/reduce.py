"""Bounding the thread: prune old turns, keep a rolling summary.

A checkpoint holds the whole message list, and DynamoDB caps a query at 1 MB.
For a text conversation that is far more headroom than a session needs - so
the answer is not to page around the limit but to make sure a thread never
approaches it. Two rules do that:

1. **Documents never enter the message list.** They go to object storage and
   only their metadata comes back as a chat message (`storage/documents.py`).
   One PDF in a transcript is worth a thousand turns of text.
2. **Old turns are reduced.** `agentstate-reducer` prunes down to a recent
   window and hands the pruned messages to a summariser, so the context that
   mattered survives as a few lines rather than as fifty messages.

Why this reducer rather than a hand-rolled one: `cascade_tool_messages`. An
AI message carrying `tool_calls` and its `ToolMessage` results are one unit -
drop the call and keep the result and Bedrock's Converse API rejects the
whole turn with "toolResult blocks exceeds toolUse blocks". A naive
"keep the last N" gets that wrong roughly half the time.

The summariser runs on the SMALL model, off the answer path, and only when
the threshold trips. Its output is marked `system_authored` so it is never
replayed to the model as the model's own prior words, and the citation
guardrail still applies to every answer composed after it - a summary is
context, never evidence.
"""
from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage

from app.config import settings
from app.obs.trace import Trace

SUMMARY_PROMPT = (
    "You are compressing an insurance conversation so it can continue "
    "without the earlier turns.\n"
    "Keep, in at most 120 words: what the customer wants, the facts they "
    "gave (vehicle, ages, city, sums insured), any quote, application or "
    "claim reference, and which step they had reached.\n"
    "Do NOT restate premiums, cover terms or decisions as if you were "
    "confirming them - this is a note to yourself, not an answer.\n"
    "Write plain prose, no headings."
)


def _summary_messages(summary: str, pruned_count: int) -> list:
    """How the summary is injected back into the surviving messages.

    The reducer injects a human/ai pair so the summary is carried in the
    message list itself rather than in a side channel the saver would not
    persist. Both are marked `system_authored`: the customer never typed the
    first, and the model never said the second.
    """
    note = (f"[context] Earlier in this conversation ({pruned_count} messages "
            f"summarised): {summary}")
    return [
        HumanMessage(content=note,
                     additional_kwargs={"system_authored": True}),
        AIMessage(content="Understood - continuing from there.",
                  additional_kwargs={"system_authored": True}),
    ]


def _summariser(trace: Trace):
    """A `summarize_fn` for the reducer, on the small model.

    Fails OPEN: if the model is unavailable the turn still proceeds, pruned
    but unsummarised. Losing older context degrades an answer; failing the
    turn denies one.
    """
    def summarise(pruned: list) -> str:
        from langchain_core.messages import SystemMessage

        from app.llm.bedrock import get_llm

        body = "\n".join(
            f"{getattr(m, 'type', '?')}: {str(getattr(m, 'content', ''))[:400]}"
            for m in pruned)
        try:
            with trace.timed("llm", "summarise", messages=len(pruned),
                             model=settings().bedrock_small_model_id):
                resp = get_llm(small=True, guardrail=False).invoke([
                    SystemMessage(content=SUMMARY_PROMPT),
                    HumanMessage(content=body[:8000])])
            content = resp.content
            if isinstance(content, list):          # Converse content blocks
                content = " ".join(b.get("text", "") for b in content
                                   if isinstance(b, dict))
            return str(content).strip()[:1200]
        except Exception as exc:                             # noqa: BLE001
            trace.add("error", "summarise_failed", detail=type(exc).__name__)
            return ""

    return summarise


def build_reducer(trace: Trace):
    from agentstate_reducer import MessageReducer, ReducerConfig

    cfg = settings()
    return MessageReducer(config=ReducerConfig(
        min_messages=cfg.reduce_keep_messages,
        max_messages=cfg.reduce_after_messages,
        # Index 0 is the oldest turn, not a system prompt - our system
        # message is prepended per call and never lives in state - so there
        # is nothing special to preserve at the front.
        preserve_first=False,
        # The one that matters: an AI message with tool_calls and its
        # ToolMessages are one unit.
        cascade_tool_messages=True,
        summarize_fn=_summariser(trace),
        inject_summary=True,
        summary_message_factory=_summary_messages,
    ))


def reduce_thread(state: dict, trace: Trace) -> dict:
    """Returns the state update that prunes the thread, or {} if not needed.

    Deletion goes through `RemoveMessage`, which is how `add_messages` is
    told to drop something - returning a shorter list would simply be merged
    back into the longer one.
    """
    cfg = settings()
    messages = list(state.get("messages") or [])
    if not cfg.reduce_enabled or len(messages) <= cfg.reduce_after_messages:
        return {}

    result = build_reducer(trace).reduce(existing=messages, new=[])
    surviving = list(result.surviving)
    kept_ids = {id(m) for m in surviving}
    removed = [m for m in messages if id(m) not in kept_ids]
    if not removed:
        return {}

    trace.add("guardrail", "thread_reduced", before=len(messages),
              after=len(surviving), pruned=len(removed),
              summarised=bool(getattr(result, "summary", None)))

    # Remove what was pruned, then add back anything the reducer injected
    # (the summary pair) that was not in the original list.
    original = {id(m) for m in messages}
    injected = [m for m in surviving if id(m) not in original]
    return {"messages": [RemoveMessage(id=m.id) for m in removed if m.id]
                        + injected}
