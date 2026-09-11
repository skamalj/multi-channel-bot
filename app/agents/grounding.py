"""Grounding: the model-based output guardrail, run once on the final answer.

Mounted as `after_agent`, so it fires exactly once per turn on the answer the
customer will see - not on every intermediate tool-calling model call. It:

1. reads the retrieval passages back out of this turn's `kb_search`
   `ToolMessage`s (no scratch state - the react loop already kept them in the
   thread),
2. gathers what earlier turns established from a source, so a follow-up that
   restates a fact is not refused for lacking a fresh citation,
3. hands both to `citations.enforce`, which strips fabricated `[n]` refs
   (deterministic) and runs the verifier (a model call - deciding whether a
   sentence is an unsupported claim is a language judgement, not a regex),
4. rewrites the final message with the enforced text, or refuses.

The heavy language work stays in `verify.py`/`citations.py`; this hook only
assembles their inputs from the message history and applies the result.
"""
from __future__ import annotations

import json
from typing import Any

from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from app.agents import citations, guardrails
from app.mcpserver.registry import get_tool
from app.obs.trace import Trace

RETRIEVAL_TOOLS = {"kb_search_health", "kb_search_motor"}


def _text_of(msg: Any) -> str:
    content = getattr(msg, "content", "")
    if isinstance(content, list):                 # Converse content blocks
        return " ".join(b.get("text", "") for b in content
                        if isinstance(b, dict)).strip()
    return (content or "").strip()


def _this_turn(messages: list) -> list:
    """The messages produced since the last human turn - this invocation's
    own model/tool work."""
    out: list = []
    for m in reversed(messages):
        if isinstance(m, HumanMessage):
            break
        out.append(m)
    return list(reversed(out))


def _payload(m: ToolMessage) -> dict:
    try:
        data = json.loads(m.content) if isinstance(m.content, str) else m.content
    except Exception:                                          # noqa: BLE001
        return {}
    return data if isinstance(data, dict) else {}


def _extra(name: str, key: str, default: str) -> str:
    t = get_tool(name)
    return ((t.extras or {}).get(key) if t else None) or default


class GroundingGuardrail(AgentMiddleware):
    def after_agent(self, state, runtime) -> dict[str, Any] | None:
        messages = list(state.get("messages") or [])
        if not messages or not isinstance(messages[-1], AIMessage):
            return None
        ai = messages[-1]
        text = _text_of(ai)
        if not text:
            return None

        trace = getattr(getattr(runtime, "context", None), "trace", None) or Trace()
        turn = _this_turn(messages)
        turn_ids = {id(m) for m in turn}

        chunks: list[dict] = []
        tools_called: list[str] = []
        used_core = False
        for m in turn:
            if not isinstance(m, ToolMessage):
                continue
            name = getattr(m, "name", "") or ""
            tools_called.append(name)
            payload = _payload(m)
            failed = "error" in payload
            if name in RETRIEVAL_TOOLS:
                chunks.extend(payload.get("chunks") or [])
            elif not failed and _extra(name, "authority", "none") == "core":
                used_core = True

        retrieval_ran = any(n in RETRIEVAL_TOOLS for n in tools_called)

        # A handoff is the safe direction and makes no factual claim - "I'll
        # put you through to a colleague" has nothing to cite and must not be
        # refused for lacking a citation. Strip fabricated refs and mask PII,
        # but do not run the claim check or the refusal.
        if "human_handoff" in tools_called:
            valid = {c["ref"] for c in chunks if "ref" in c}
            text2, invented = citations.strip_unknown_refs(text, valid)
            text2 = guardrails.mask_pii(text2)
            trace.add("guardrail", "citations", cited=[], dropped=[],
                      invented_refs=invented, verifier="skipped_handoff",
                      refused=False, grounded_by="handoff", tools=tools_called)
            update = {"last_tools": tools_called, "last_citations": []}
            if text2 != text:
                update["messages"] = [AIMessage(content=text2 or "(no reply)",
                                                id=ai.id)]
            return update

        # human_handoff aside, a state change (a write/dispatch that makes a
        # claim) is what picks the action-shaped refusal wording.
        action_attempted = any(
            _extra(n, "effect", "read") in ("write", "dispatch")
            for n in tools_called if n != "human_handoff")

        enforced, report = citations.enforce(
            text, chunks, other_tool_evidence=used_core,
            established=_established(messages, turn_ids),
            retrieval_ran=retrieval_ran, action_attempted=action_attempted)

        # The one deterministic guardrail, at the output boundary: mask a full
        # Aadhaar or card number the model may have echoed despite the prompt.
        enforced = guardrails.mask_pii(enforced)

        trace.add("guardrail", "citations",
                  cited=sorted(set(report.get("cited", []))),
                  dropped=report.get("dropped", []),
                  invented_refs=report.get("invented_refs", []),
                  verifier=report.get("verifier"),
                  refused=report.get("refused", False),
                  grounded_by=("retrieval" if chunks
                               else "core tool" if used_core else "nothing"),
                  tools=tools_called)

        update: dict[str, Any] = {
            "last_tools": tools_called,
            "last_citations": sorted(set(report.get("cited", []))),
        }
        if enforced != text:
            # Same id replaces the model's message in place, so the transcript
            # holds exactly what the customer was shown.
            update["messages"] = [AIMessage(content=enforced or "(no reply)",
                                            id=ai.id)]
        return update


def _established(messages: list, turn_ids: set) -> list[str]:
    """What EARLIER turns established from a source - core-tool facts and the
    passages that were retrieved before this turn. These live in the thread's
    ToolMessages, so a follow-up ("how many months was that again?") can be
    checked against them instead of being refused for want of a fresh
    citation."""
    facts: list[str] = []
    for m in messages:
        if id(m) in turn_ids or not isinstance(m, ToolMessage):
            continue
        name = getattr(m, "name", "") or ""
        payload = _payload(m)
        if "error" in payload:
            continue
        if name in RETRIEVAL_TOOLS:
            for c in (payload.get("chunks") or []):
                t = str(c.get("text") or "")[:300]
                if t:
                    facts.append(t)
        elif _extra(name, "authority", "none") == "core" and payload:
            facts.append(json.dumps(payload, default=str)[:400])
    return facts[-12:]
