"""LLM binding.

Bedrock through the Converse API (`langchain-aws`), which is why nothing here
is vendor-specific: the agent loop runs on Kimi K2.5, Nova, Claude or
anything else Bedrock exposes with `toolConfig`, and the model id is
configuration.

Two small-model helpers live here as well. Both are deliberately off the
answer path: an intent classification that only runs when the deterministic
signals produced nothing, and a reranker that is the drop-in for a real
cross-encoder. Both fail CLOSED - a classifier that errors returns "no
opinion", which routes to asking the customer, and never to a guess.

`MOCK_LLM=1` swaps in a scripted stub so the whole pipeline - resolver,
binding, tools, gates, citations - runs with no credentials at all.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any

from app.config import settings

log = logging.getLogger("mcb.llm")


# ---------------------------------------------------------------------------
# Scripted stub (NF-2)
# ---------------------------------------------------------------------------
class StubLLM:
    """Deterministic stand-in that drives the SAME paths as a real model.

    It picks a tool when the text asks for one, and composes a cited answer
    from the tool result on the next round - so the citation guardrail, the
    refusal path and the tool-round limit are all exercised without Bedrock.
    """

    def __init__(self, tools: list[Any] | None = None):
        self.tools = tools or []

    def bind_tools(self, tools):
        return StubLLM(tools)

    @property
    def _names(self) -> set[str]:
        return {t["name"] if isinstance(t, dict) else getattr(t, "name", "")
                for t in self.tools}

    def invoke(self, messages, **_):
        from langchain_core.messages import AIMessage

        last = messages[-1]
        if getattr(last, "type", "") == "tool":
            return AIMessage(content=self._compose(messages))

        text = ""
        for m in reversed(messages):
            if getattr(m, "type", "") == "human":
                text = m.content
                break
        call = self._choose(text or "")
        if call:
            return AIMessage(content="", tool_calls=[call])
        return AIMessage(content=f"[stub] {text[:200]}")

    def _choose(self, text: str) -> dict | None:
        low = text.lower()
        n = self._names

        def call(name, args):
            return {"name": name, "id": f"stub-{name}", "args": args}

        if "commission" in low and "commission_statement" in n:
            return call("commission_statement",
                        {"producer_id": "P-2201", "period": "2026-08"})
        if any(k in low for k in ("handoff", "human", "agent please")) \
                and "human_handoff" in n:
            return call("human_handoff", {"reason": "customer asked",
                                          "summary": text[:200]})
        if "quote" in low or "premium" in low:
            if "quote_create_health" in n:
                return call("quote_create_health",
                            {"product_id": "PHS", "sum_insured": 1000000,
                             "member_ages": [34, 37], "city": "Pune",
                             "addons": []})
            if "quote_create_motor" in n:
                return call("quote_create_motor",
                            {"product_id": "PMS", "idv": 480000, "ncb_pct": 35,
                             "vehicle_age_years": 4, "addons": []})
        m = re.search(r"\b([A-Z]{2}\s?\d{1,2}\s?[A-Z]{1,3}\s?\d{4})\b", text)
        if m and "vehicle_lookup" in n:
            return call("vehicle_lookup", {"registration": m.group(1)})
        if "waiting period" in low and "member_waiting_periods" in n \
                and re.search(r"\bmy\b", low):
            return call("member_waiting_periods", {"policy_id": "PHS-4471902"})
        for kb in ("kb_search_health", "kb_search_motor"):
            if kb in n:
                return call(kb, {"query": text})
        return None

    def _compose(self, messages) -> str:
        """Answer from the tool result, citing chunk ids - which is exactly
        what the citation guardrail then checks."""
        payload: Any = None
        for m in reversed(messages):
            if getattr(m, "type", "") == "tool":
                try:
                    payload = json.loads(m.content)
                except Exception:                            # noqa: BLE001
                    payload = m.content
                break
        if isinstance(payload, dict) and "chunks" in payload:
            chunks = payload["chunks"]
            if not chunks:
                return ("I could not find an approved source that answers "
                        "that. I would rather hand you to a colleague than "
                        "guess.")
            top = chunks[0]
            return (f"{top['text']} [{top['chunk_id']}] "
                    f"(Source: {top['source']}, {top['section']}, "
                    f"page {top['page']}.)")
        if isinstance(payload, dict) and payload.get("error"):
            return (f"That did not go through: {payload.get('detail') or payload['error']}. "
                    f"Shall I put you through to a colleague?")
        if isinstance(payload, dict) and "gross_premium" in payload:
            return (f"[stub] {payload.get('product_name')}: premium "
                    f"Rs {payload['gross_premium']:,.0f} including GST "
                    f"(quote {payload.get('quote_id')}, valid to "
                    f"{payload.get('valid_until')}).")
        return f"[stub] {json.dumps(payload, default=str)[:400]}"


# ---------------------------------------------------------------------------
# Bedrock
# ---------------------------------------------------------------------------
def get_llm(small: bool = False, guardrail: bool = True):
    """The model. `guardrail=False` is for calls a customer never reads.

    The guardrail exists to protect what a customer sees, and it belongs on
    the call that writes to them. It does not belong on the internal calls
    that decide things ABOUT that answer - the claim verifier, the thread
    summariser, the intent classifier, the reranker. Every one of those reads
    the model's reply back as JSON, and an intervention does not raise: it
    replaces the reply with the block message. So the JSON fails to parse and
    the check silently does not happen.

    That was not hypothetical. The verifier returned `unparseable_verdict` on
    a live turn because the guardrail intervened on the VERIFIER's own call -
    the one deciding whether the answer was safe to send. A safety control
    that disables the safety machinery is worse than not having it there.

    Nothing is weakened by this: the customer-facing generation still carries
    it, and so does the inbound screen, which is where untrusted text arrives.
    """
    cfg = settings()
    if cfg.mock_llm:
        return StubLLM()
    from langchain_aws import ChatBedrockConverse

    from app.agents.guardrail import model_config

    kwargs: dict[str, Any] = dict(
        model=cfg.bedrock_small_model_id if small else cfg.bedrock_model_id,
        region_name=cfg.aws_region,
        temperature=cfg.llm_temperature,
        max_tokens=cfg.llm_max_tokens,
    )
    # The guardrail rides on the Converse call itself. When it intervenes the
    # response carries stopReason == "guardrail_intervened" rather than
    # raising, so callers check `guardrail.intervened(resp)` - see the model
    # node. None here means no guardrail is deployed, which is the offline
    # posture, not a silent opt-out.
    gc = model_config() if guardrail else None
    if gc:
        kwargs["guardrail_config"] = gc
    return ChatBedrockConverse(**kwargs)


def _json_from(text: str) -> dict:
    """Models wrap JSON in prose and fences. Take the first object, or fail."""
    if not text:
        return {}
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return {}
    try:
        return json.loads(m.group(0))
    except Exception:                                        # noqa: BLE001
        return {}


_INTENT_PROMPT = """You classify one message from an insurance customer or \
agent into a line of business.

Allowed values: {options}, or "unknown".

Rules:
- Answer "unknown" unless the message itself gives you evidence. A greeting, \
a thank-you or a general question is "unknown".
- Do not guess from what is statistically common.
- confidence is your probability that the label is right, 0.0 to 1.0.

Reply with JSON only: {{"lob": "...", "confidence": 0.0, "evidence": "the \
words that decided it"}}"""


def classify_lob(text: str, options: list[str]) -> tuple[str | None, float, str]:
    """RS-3's intent model. The LAST resort before asking, and it fails closed.

    An exception, an unparseable reply or a label outside `options` all give
    (None, 0.0) - which routes to a question, never to a guess.
    """
    cfg = settings()
    if not cfg.intent_model_enabled or not (text or "").strip():
        return None, 0.0, ""
    if cfg.mock_llm:
        return None, 0.0, "intent model disabled under MOCK_LLM"
    try:
        from langchain_core.messages import HumanMessage, SystemMessage

        llm = get_llm(small=True, guardrail=False)
        resp = llm.invoke([
            SystemMessage(content=_INTENT_PROMPT.format(
                options=", ".join(f'"{o}"' for o in options))),
            HumanMessage(content=text[:1000]),
        ])
        content = resp.content
        if isinstance(content, list):                # Converse block form
            content = " ".join(b.get("text", "") for b in content
                               if isinstance(b, dict))
        data = _json_from(str(content))
        lob = data.get("lob")
        conf = float(data.get("confidence", 0.0) or 0.0)
        if lob not in options:
            return None, 0.0, str(data.get("evidence", ""))[:120]
        return lob, max(0.0, min(conf, 1.0)), str(data.get("evidence", ""))[:120]
    except Exception as exc:                                 # noqa: BLE001
        log.warning("intent model unavailable, falling back to asking: %s", exc)
        return None, 0.0, f"intent model unavailable: {type(exc).__name__}"


_RERANK_PROMPT = """Score how well each passage answers the question.
0.0 = irrelevant, 1.0 = directly answers it.
Reply with JSON only: {"scores": {"<chunk_id>": 0.0}}"""


def score_relevance(query: str, passages: list[dict]) -> dict[str, float]:
    """Optional LLM reranker - the drop-in for a real cross-encoder."""
    if settings().mock_llm or not passages:
        return {}
    from langchain_core.messages import HumanMessage, SystemMessage

    body = "\n\n".join(f"[{p['chunk_id']}] {p['text']}" for p in passages)
    resp = get_llm(small=True, guardrail=False).invoke([
        SystemMessage(content=_RERANK_PROMPT),
        HumanMessage(content=f"Question: {query}\n\nPassages:\n{body}"),
    ])
    content = resp.content
    if isinstance(content, list):
        content = " ".join(b.get("text", "") for b in content
                           if isinstance(b, dict))
    scores = _json_from(str(content)).get("scores", {})
    return {k: float(v) for k, v in scores.items()
            if isinstance(v, (int, float))}
