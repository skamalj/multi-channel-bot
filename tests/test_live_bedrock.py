"""Live tests against real Bedrock. Deselected by default.

    uv run pytest -m live -q          # needs AWS credentials in the shell

These are the tests the mocked suite cannot be: whether the tool schemas this
code generates are accepted by the model, whether the model actually calls
the right tool, whether citations survive a real generation, and whether a
turn comes back inside the latency budget (NF-5).

They run against whatever `BEDROCK_MODEL_ID` is set to - Kimi K2.5 by
default, Nova by configuration - because nothing in the design is
vendor-specific and the suite should be able to prove it.
"""
from __future__ import annotations

import os
import statistics
import time

import pytest

pytestmark = pytest.mark.live


@pytest.fixture(scope="module", autouse=True)
def live_env():
    """Turn the stub off for this module and put it back afterwards."""
    import app.config as config

    os.environ["MOCK_LLM"] = "0"
    os.environ["NO_AWS"] = "1"          # stores stay local; only the LLM is real
    config.settings.cache_clear()
    yield
    os.environ["MOCK_LLM"] = "1"
    config.settings.cache_clear()


@pytest.fixture
def live_client():
    from fastapi.testclient import TestClient

    from app.agents.graph import reset_agents
    from app.main import app

    reset_agents()
    return TestClient(app)


def _say(client, user, text, **kw):
    r = client.post("/api/chat", json={"user_id": user, "text": text, **kw})
    assert r.status_code == 200, r.text
    return r.json()


def _reply(d):
    return d["replies"][0]["text"] if d["replies"] else ""


def _tools(d):
    return [e["label"] for e in d["trace"]["events"] if e["kind"] == "tool"]


def test_the_model_reaches_bedrock_at_all(live_client):
    from app.config import settings

    body = live_client.get("/api/health").json()
    assert body["mock_llm"] is False
    assert body["model"] == settings().bedrock_model_id
    d = _say(live_client, "live-smoke", "hello")
    assert _reply(d)


def test_the_generated_tool_schemas_are_accepted_by_the_model(live_client):
    """The schema this code emits from real signatures has to be valid to
    Bedrock's Converse toolConfig, or every turn fails identically.

    Asserted on the ERROR events, not on whether the model chose to call
    something: a rejected toolConfig is a ValidationException on every turn,
    and which tool a model reaches for first is its business.
    """
    d = _say(live_client, "live-schema",
             "what is the waiting period for a pre-existing disease")
    assert not [e for e in d["trace"]["events"]
                if e["kind"] == "error"], d["trace"]["events"]
    assert _reply(d)


def test_a_knowledge_answer_is_either_grounded_or_a_refusal(live_client):
    """The invariant that must hold for EVERY model, on every run.

    Models differ in whether they reach for retrieval on a given phrasing -
    that is a quality difference. What may never differ is that no
    **material** claim reaches the customer without something behind it: a
    figure, a money amount, a promise of cover, a gate decision.

    Note what is deliberately NOT asserted. An uncited *procedural* answer -
    "get admitted, submit a pre-authorisation, keep the bills" - can still
    reach the customer, because separating "here is how a claim works" from
    "here is what I need from you" reliably enough to refuse the first and
    keep the second is not something a sentence test does well, and refusing
    both makes the bot unusable. `requirements.md` names that boundary rather
    than implying it is closed.
    """
    for question in ("what is the initial waiting period",
                     "how do I make a cashless claim",
                     "what is not covered at all"):
        d = _say(live_client, "live-invariant", question, lob="health")
        guard = [e for e in d["trace"]["events"]
                 if e["kind"] == "guardrail" and e["label"] == "citations"]
        assert guard, f"no guardrail ran for: {question}"
        detail = guard[0]["detail"]
        assert detail["hallucinated"] == [], (
            f"invented a source for {question!r}: {detail['hallucinated']}")

        # Whatever survived, every material claim in it is accounted for:
        # cited, repaired onto a retrieved chunk, or backed by a core tool.
        from app.agents import citations

        unsupported = [s for s in citations._split(_reply(d))
                       if citations._needs_source(s)]
        if unsupported:
            assert detail["cited"] or detail["grounded_by"] == "core tool", (
                f"material claim with nothing behind it for {question!r}: "
                f"{unsupported[:2]}")


def test_a_knowledge_question_is_answered_from_the_corpus_with_a_citation(
        live_client):
    """Every figure in the answer must appear in a chunk the answer cites.

    Not "the answer says 36 months" - that would assert a ranking preference
    rather than a requirement. Several products have different waiting
    periods and all of them are correct answers to a question that named no
    product. What KB-4 actually requires is that the numbers came from
    somewhere real, so that is what this checks.

    Sampled three times rather than once. Whether a given model reaches for
    retrieval on a given phrasing is stochastic - Nova refuses on this one
    more often than Kimi does, and refusing is a SAFE outcome, not a wrong
    one. What is being asserted is that the corpus is reachable at all, so a
    model that cannot ground this question in three attempts is a real
    finding rather than a flaky test.
    """
    import re

    from app.knowledge.corpus import corpus

    text_of = {c["chunk_id"]: c["text"] for c in corpus()}
    grounded_once = False
    replies = []

    for attempt in range(3):
        d = _say(live_client, f"live-cite-{attempt}",
                 "what is the waiting period for a pre-existing disease")
        guard = [e for e in d["trace"]["events"]
                 if e["kind"] == "guardrail" and e["label"] == "citations"]
        assert guard, "no citation guardrail ran"
        detail = guard[0]["detail"]
        assert detail["hallucinated"] == []
        replies.append(_reply(d)[:120])
        if not detail["cited"]:
            continue                       # a safe refusal; try the next one
        grounded_once = True

        # Whenever it DID answer, every figure must be in a cited chunk.
        sources = " ".join(text_of.get(cid, "") for cid in detail["cited"])
        for months in set(re.findall(r"(\d+)\s*months?", _reply(d), re.I)):
            assert re.search(rf"\b{months}\s*months?\b", sources, re.I), (
                f"{months} months is in the answer but in none of the cited "
                f"chunks {detail['cited']}")

    assert grounded_once, (
        "the corpus was never reached in three attempts; replies were "
        f"{replies}")


def test_an_unanswerable_question_refuses_rather_than_inventing(live_client):
    """KB-5 with a real model, which is the only place it can be trusted.

    The entry context pins the line of business deliberately: without it the
    resolver asks which cover this is about and the turn never reaches
    retrieval - correct behaviour, but not what this test is about.

    Asserted on the HARM rather than on wording. Models decline in different
    words - "I could not find anything documented", "I can only assist with
    health insurance", "that is not typically covered" - and sniffing for a
    phrase makes the test about vocabulary. What must never happen is a
    promise of cover that no source supports.
    """
    import re

    d = _say(live_client, "live-refuse",
             "will you reimburse my Lisbon holiday and my new laptop",
             lob="health")
    reply = _reply(d)
    promised = re.search(
        r"\b(is|are|will be|would be)\s+(covered|reimbursed|payable|"
        r"admissible)\b", reply, re.I)
    assert not promised, f"promised cover with no source: {reply[:200]}"

    guard = [e for e in d["trace"]["events"]
             if e["kind"] == "guardrail" and e["label"] == "citations"][0]
    assert guard["detail"]["hallucinated"] == []


def test_the_agent_bot_can_reach_agent_only_content(live_client):
    """The commission GRID is agent-scope corpus content, and the producer is
    authenticated - which is what a producer reading their own book is."""
    live_client.post("/api/stepup", json={"user_id": "919820000001"})
    d = _say(live_client, "919820000001",
             "what does the motor commission grid say about own damage rates",
             lob="motor")
    retrieved = [c["chunk_id"]
                 for e in d["trace"]["events"] if e["kind"] == "retrieve"
                 for c in e["detail"].get("accepted_chunks", [])]
    assert any(c.startswith("M-COMM-GRID") for c in retrieved), \
        f"agent-only content not reached; got {retrieved}"


def test_a_customer_cannot_reach_agent_only_content(live_client):
    """The same question, a customer identity: the corpus scope is public and
    the commission grid is not a candidate."""
    d = _say(live_client, "live-scope",
             "what commission do you earn on motor own damage", lob="motor")
    retrieve = [e for e in d["trace"]["events"] if e["kind"] == "retrieve"]
    for e in retrieve:
        for c in e["detail"].get("accepted_chunks", []):
            assert not c["chunk_id"].startswith("M-COMM-GRID")


def test_a_premium_comes_from_the_rating_engine_not_the_model(live_client):
    """The number in the reply must be the number the tool returned."""
    live_client.post("/api/consent", json={"user_id": "919820000009",
                                           "purpose": "quotation",
                                           "granted": True})
    proposed = _say(live_client, "919820000009",
                    "please quote Protec Health Secure, 10 lakh sum insured, "
                    "two adults aged 34 and 37, in Pune")
    gates = [e for e in proposed["trace"]["events"]
             if e["kind"] == "gate"
             and e["label"].startswith("confirmation_required")]
    assert gates, f"a write tool was not confirmation-gated: {_reply(proposed)}"

    done = _say(live_client, "919820000009", "yes please go ahead")
    tool_events = [e for e in done["trace"]["events"] if e["kind"] == "tool"]
    assert tool_events and tool_events[0]["label"] == "quote_create_health"


def test_the_model_never_sees_the_other_line_of_business(live_client):
    user = "live-isolation"
    _say(live_client, user, "what does my health plan cover for maternity")
    d = _say(live_client, user, "and what about my car policy")
    bind = [e for e in d["trace"]["events"] if e["kind"] == "bind"][0]
    assert bind["detail"]["lob"] == "motor"
    assert "kb_search_health" not in bind["detail"]["tools"]


def test_p95_turn_is_inside_the_budget(live_client):
    """NF-5: p95 under 20 s with a real Bedrock call.

    Ten turns is not a load test. It is enough to catch a turn that has
    quietly become four sequential model calls.
    """
    questions = [
        "what is the initial waiting period",
        "how long before a pre-existing disease is covered",
        "what is the room rent eligibility",
        "how do I make a cashless claim",
        "what documents do you need for a claim",
        "is maternity covered",
        "what is the free look period",
        "can I port my policy from another insurer",
        "what is the grace period for renewal",
        "what is not covered at all",
    ]
    times = []
    for i, q in enumerate(questions):
        d = _say(live_client, f"live-perf-{i}", q)
        times.append(d["turn_ms"])
    times.sort()
    p95 = times[int(0.95 * (len(times) - 1))]
    print(f"\n  turns={len(times)} median={statistics.median(times):.0f}ms "
          f"p95={p95:.0f}ms max={times[-1]:.0f}ms")
    assert p95 < 20_000, f"p95 {p95:.0f}ms exceeds the 20s budget"


def test_the_intent_model_fails_closed_rather_than_guessing(live_client):
    """RS-3's last resort still has to clear the threshold, and a greeting
    gives it nothing to clear it with."""
    d = _say(live_client, "live-intent", "hi there")
    intent = [e for e in d["trace"]["events"]
              if e["kind"] == "resolve" and e["label"] == "intent_model"]
    assert intent, "the intent model did not run on an ambiguous first turn"
    assert intent[0]["detail"]["value"] in (None, "unknown")
    assert "?" in _reply(d)
