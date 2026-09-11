"""LLM binding.

Bedrock through the Converse API (`langchain-aws`), which is why nothing here
is vendor-specific: the agent loop runs on Kimi K2.5, Nova, Claude or
anything else Bedrock exposes with `toolConfig`, and the model id is
configuration.

The small judgements live here too, and they are here rather than scattered
because each one replaced a pattern that got language wrong:

* `classify_lob` routes a turn to health or motor. It used to be a keyword
  list at 0.95 confidence with the model kept as a last resort.
* `score_relevance` is the drop-in for a real cross-encoder.

`classify_lob` fails to "no opinion", which routes to a question rather than
a guess.

`MOCK_LLM=1` swaps in a scripted stub so the whole pipeline - resolver,
binding, tools, gates, citations - runs with no credentials at all.
"""
from __future__ import annotations

import json
import logging
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

    def bind_tools(self, tools, **_):
        # create_agent binds with tool_choice=... and model_settings; the stub
        # ignores them and just remembers the tool set.
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
        reg = _registration_like(text)
        if reg and "vehicle_lookup" in n:
            return call("vehicle_lookup", {"registration": reg})
        if "waiting period" in low and "member_waiting_periods" in n \
                and " my " in f" {low} ":
            return call("member_waiting_periods", {"policy_id": "PHS-4471902"})
        for kb in ("kb_search_health", "kb_search_motor"):
            if kb in n:
                return call(kb, {"query": text})
        return None

    def _compose(self, messages) -> str:
        """Answer from the tool result, citing the passage NUMBER - which is
        what the retrieval tool asks the real model for, and what the citation
        guardrail then checks.

        This cited the chunk id until the regexes came out of citations.py,
        and passed only because the old pattern happened to find a digit
        inside `PHS-POLICY_WORDING-V2#1`. A stub that exercises a path no real
        model takes is worse than no stub.
        """
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
            return (f"{top['text']} [{top.get('ref', 1)}] "
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
def get_llm(small: bool = False):
    """The model, and nothing else.

    Guardrails are NOT here. In this framework a guardrail is provider-agnostic
    middleware in the agent's middleware list (app/agents/guardrails.py), not a
    property of the model object. Tying screening to `ChatBedrockConverse`'s
    own `guardrails=` parameter would lock it to one vendor and put it outside
    the middleware system the framework defines. So the model is just a model;
    the input guardrail and the grounding guardrail run as hooks around it, and
    the same model serves the internal calls (verifier, summariser, classifier)
    that a customer never reads.
    """
    cfg = settings()
    if cfg.mock_llm:
        return StubLLM()
    from langchain_aws import ChatBedrockConverse

    return ChatBedrockConverse(
        model=cfg.bedrock_small_model_id if small else cfg.bedrock_model_id,
        region_name=cfg.aws_region,
        temperature=cfg.llm_temperature,
        max_tokens=cfg.llm_max_tokens,
    )


def _json_from(text: str) -> dict:
    """Models wrap JSON in prose and fences. Take the first object, or fail.

    Brace-to-brace rather than a pattern: finding the outermost object is
    counting characters, not recognising a shape.
    """
    if not text:
        return {}
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return {}
    try:
        return json.loads(text[start:end + 1])
    except Exception:                                        # noqa: BLE001
        return {}


_INTENT_PROMPT = """You classify one message from an insurance customer or \
agent into a line of business.

Allowed values: {options}, or "unknown".

Our products, which a customer will name without saying which line they are:
{products}

Rules:
- Naming one of our products IS evidence. "What is the waiting period on \
Health Secure" is health, because Health Secure is a health product.
- Otherwise answer "unknown" unless the message itself gives you evidence. A \
greeting, a thank-you or a general question is "unknown".
- Do not guess from what is statistically common.
- confidence is your probability that the label is right, 0.0 to 1.0.

Reply with JSON only: {{"lob": "...", "confidence": 0.0, "evidence": "the \
words that decided it"}}"""


def _product_vocabulary(options: list[str]) -> str:
    """Our product names, per line of business, for the classifier.

    Not a keyword list deciding anything - the model still decides. This is
    the vocabulary it cannot be expected to have: "Health Secure" means
    nothing to a general model, and it read "the waiting period on Health
    Secure" as a question that "does not specify a line of business". So
    every conversation that named a product by name was asked which line of
    business it was about, over and over, and never got past the question.

    Read from the catalogue, so a product added there is a product the router
    recognises without anyone remembering to update a list.
    """
    from app.coremock.catalog import catalog

    book = catalog()
    lines = []
    for lob in options:
        names = [f"{p.get('name')} ({p.get('product_id')})"
                 for p in book.get(lob, []) if p.get("name")]
        if names:
            lines.append(f"- {lob}: " + ", ".join(names))
    return "\n".join(lines) or "- (no catalogue available)"


def classify_lob(text: str, options: list[str]) -> tuple[str | None, float, str]:
    """RS-3's intent model. The LAST resort before asking, and it fails closed.

    An exception, an unparseable reply or a label outside `options` all give
    (None, 0.0) - which routes to a question, never to a guess.
    """
    cfg = settings()
    if not cfg.intent_model_enabled or not (text or "").strip():
        return None, 0.0, ""
    if cfg.mock_llm or cfg.no_aws:
        return _stub_lob(text, options)
    try:
        from langchain_core.messages import HumanMessage, SystemMessage

        llm = get_llm(small=True)
        resp = llm.invoke([
            SystemMessage(content=_INTENT_PROMPT.format(
                options=", ".join(f'"{o}"' for o in options),
                products=_product_vocabulary(options))),
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
    resp = get_llm(small=True).invoke([
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


def _registration_like(text: str) -> str | None:
    """A vehicle registration in the stub's input, found by counting.

    Stub logic: it only has to recognise the seeded registrations so the
    offline pipeline reaches vehicle_lookup. It validates nothing, and
    nothing deployed calls it.
    """
    for token in (text or "").replace("-", " ").split():
        core = token.strip(".,;:!?()")
        if not core.isalnum() or not core[:1].isalpha():
            continue
        letters = sum(c.isalpha() for c in core)
        digits = sum(c.isdigit() for c in core)
        if letters >= 2 and digits >= 4:
            return core
    return None


# The offline stand-in for classify_lob. Keyword membership, and it is a STUB -
# it exists so the pipeline runs with no credentials, in the same way StubLLM
# stands in for the model itself. The deployed path never reaches it. The
# routing rule it replaced was keyword matching in the resolver, on the live
# path, at 0.95 confidence.
_STUB_LOB_WORDS = {
    "motor": ("car", "bike", "scooter", "vehicle", "motor", "idv", "ncb",
              "garage", "windscreen", "bumper"),
    "health": ("health", "medical", "hospital", "mediclaim", "cashless",
               "maternity", "opd", "waiting period", "sum insured",
               "pre-existing", "room rent"),
}


def _stub_lob(text: str, options: list[str]) -> tuple[str | None, float, str]:
    low = (text or "").lower()
    hit = {lob: [w for w in words if w in low]
           for lob, words in _STUB_LOB_WORDS.items() if lob in options}
    named = {lob: words for lob, words in hit.items() if words}
    if len(named) != 1:
        return None, 0.0, "stub: nothing decisive"
    lob, words = next(iter(named.items()))
    return lob, 0.95, words[0]
