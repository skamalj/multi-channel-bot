"""Bedrock Guardrails, applied in the two places the ecosystem applies them.

LangChain splits this deliberately, and the split is the interesting part:

1. **On the model.** `ChatBedrockConverse(guardrail_config=...)` runs the
   guardrail over the prompt and the completion inside the Converse call. When
   it intervenes the response comes back with
   `response_metadata["stopReason"] == "guardrail_intervened"` rather than an
   exception - so it has to be checked for, or a blocked turn quietly becomes
   an empty answer.

2. **As its own step.** `ApplyGuardrail` is a direct API call over text the
   model never generated: retrieved passages, tool results, an inbound
   message from a channel. The model guardrail never sees those on their own,
   and they are exactly where a poisoned document or a compromised upstream
   would arrive.

The second is also where **contextual grounding** lives, because grounding
needs the source passages and the query as separate inputs - it is asking
"is this answer supported by THIS material", which is not a question the
model-level filter can pose.

**What this does not replace.** `app/agents/citations.py` stays. Grounding is
a score; the citation rule is a rule. A fabricated premium that reads as
well-grounded clears a threshold, and still fails "name the document you got
that from". They fail differently on purpose.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from app.config import settings

log = logging.getLogger("mcb.guardrail")

INTERVENED = "guardrail_intervened"


@dataclass(frozen=True)
class GuardrailVerdict:
    """What a guardrail said, in terms the graph can branch on."""

    blocked: bool
    action: str = "NONE"                    # NONE | GUARDRAIL_INTERVENED
    reasons: list[str] = field(default_factory=list)
    text: str | None = None                 # masked/rewritten text, if any
    grounding: float | None = None
    relevance: float | None = None

    @property
    def ok(self) -> bool:
        return not self.blocked


def configured() -> bool:
    cfg = settings()
    return bool(cfg.guardrail_id) and not cfg.mock_llm


def model_config() -> dict | None:
    """The `guardrail_config` to hand ChatBedrockConverse.

    Returns None when no guardrail is deployed, so the agent runs unchanged
    offline and in tests. A guardrail that is not configured must not become
    a guardrail that silently fails open in production - that is what
    `configured()` is checked against at bind time, and what the deploy
    asserts.
    """
    if not configured():
        return None
    cfg = settings()
    return {
        "guardrailIdentifier": cfg.guardrail_id,
        "guardrailVersion": cfg.guardrail_version,
        # `enabled` returns the assessment alongside the response, which is
        # what lets the trace say WHY a turn was blocked rather than just
        # that it was.
        "trace": "enabled",
    }


def intervened(response: Any) -> bool:
    """Did the model-level guardrail stop this response?

    Checked explicitly rather than inferred from an empty completion: a
    blocked turn and a model that simply had nothing to say are different
    events and must not be treated alike.
    """
    meta = getattr(response, "response_metadata", None) or {}
    return meta.get("stopReason") == INTERVENED


# ---------------------------------------------------------------------------
# ApplyGuardrail - for text the model did not generate
# ---------------------------------------------------------------------------
def _client():
    import boto3

    return boto3.client("bedrock-runtime", region_name=settings().aws_region)


def _assessment(resp: dict) -> GuardrailVerdict:
    action = resp.get("action", "NONE")
    reasons: list[str] = []
    grounding = relevance = None

    for a in resp.get("assessments") or []:
        for f in (a.get("contentPolicy") or {}).get("filters") or []:
            if f.get("action") == "BLOCKED":
                reasons.append(f"content:{f.get('type', '?').lower()}")
        for t in (a.get("topicPolicy") or {}).get("topics") or []:
            if t.get("action") == "BLOCKED":
                reasons.append(f"topic:{t.get('name', '?')}")
        for w in (a.get("wordPolicy") or {}).get("customWords") or []:
            if w.get("action") == "BLOCKED":
                reasons.append("word:custom")
        for m in (a.get("wordPolicy") or {}).get("managedWordLists") or []:
            if m.get("action") == "BLOCKED":
                reasons.append(f"word:{m.get('type', '?').lower()}")
        pii = (a.get("sensitiveInformationPolicy") or {})
        for p in pii.get("piiEntities") or []:
            if p.get("action") in ("BLOCKED", "ANONYMIZED"):
                reasons.append(f"pii:{p.get('type', '?').lower()}")
        for r in pii.get("regexes") or []:
            if r.get("action") in ("BLOCKED", "ANONYMIZED"):
                reasons.append(f"regex:{r.get('name', '?')}")
        for g in (a.get("contextualGroundingPolicy") or {}).get("filters") or []:
            if g.get("type") == "GROUNDING":
                grounding = g.get("score")
            elif g.get("type") == "RELEVANCE":
                relevance = g.get("score")
            if g.get("action") == "BLOCKED":
                reasons.append(f"grounding:{g.get('type', '?').lower()}")

    text = None
    outputs = resp.get("outputs") or []
    if outputs:
        text = outputs[0].get("text")

    return GuardrailVerdict(
        blocked=action == "GUARDRAIL_INTERVENED",
        action=action, reasons=reasons, text=text,
        grounding=grounding, relevance=relevance)


def apply(text: str, *, source: str = "INPUT",
          grounding_source: str | None = None,
          query: str | None = None) -> GuardrailVerdict:
    """Run the guardrail over text directly.

    `source` is OUTPUT for anything the system is about to say and INPUT for
    anything arriving from outside. Pass `grounding_source` and `query` to
    engage the contextual grounding check - without both, grounding has
    nothing to ground against and is simply not evaluated.

    **Fails open, loudly.** If the guardrail API is unreachable the turn
    proceeds and the trace records it. That is the right trade here only
    because the citation guardrail still runs afterwards and is not network
    dependent - this is a second lock on the same door, not the only one.
    """
    if not configured():
        return GuardrailVerdict(blocked=False, action="NOT_CONFIGURED")

    cfg = settings()
    content: list[dict] = []
    if grounding_source and query:
        content.append({"text": {"text": grounding_source,
                                 "qualifiers": ["grounding_source"]}})
        content.append({"text": {"text": query, "qualifiers": ["query"]}})
    content.append({"text": {"text": text}})

    try:
        resp = _client().apply_guardrail(
            guardrailIdentifier=cfg.guardrail_id,
            guardrailVersion=cfg.guardrail_version,
            source=source,
            content=content)
    except Exception as exc:                                 # noqa: BLE001
        log.warning("apply_guardrail failed: %s: %s", type(exc).__name__, exc)
        return GuardrailVerdict(blocked=False, action="ERROR",
                                reasons=[type(exc).__name__])
    return _assessment(resp)


def screen_inbound(text: str) -> GuardrailVerdict:
    """An inbound message, before it reaches the model.

    A channel is an untrusted boundary. WhatsApp in particular delivers text
    written by anyone who knows the number.
    """
    return apply(text, source="INPUT")


def screen_retrieved(passages: list[str], query: str,
                     answer: str) -> GuardrailVerdict:
    """The answer, against the passages it was supposed to come from.

    This is the call the model-level guardrail cannot make, because it never
    sees the retrieved material as a separate thing from the prompt.
    """
    return apply(answer, source="OUTPUT",
                 grounding_source="\n\n".join(p for p in passages if p),
                 query=query)
