"""End-to-end through the real HTTP surface - the same requests the web
console makes.

These are not unit tests with a mocked orchestrator. Each one drives
`POST /api/chat` and asserts on the reply and on the trace the console
renders, which means a regression in wiring shows up here rather than in a
demo.
"""
from __future__ import annotations

CUSTOMER = "919820000009"          # holds a health AND a motor policy
HEALTH_ONLY = "919820000002"
PRODUCER = "919820000001"
STRANGER = "919877000123"


def _events(data, kind=None, label=None):
    out = data["trace"]["events"]
    if kind:
        out = [e for e in out if e["kind"] == kind]
    if label:
        out = [e for e in out if e["label"] == label]
    return out


def _say(client, user, text, **kw):
    r = client.post("/api/chat", json={"user_id": user, "text": text, **kw})
    assert r.status_code == 200, r.text
    return r.json()


def _reply(data):
    return data["replies"][0]["text"] if data["replies"] else ""


# --- surface ---------------------------------------------------------------
def test_health_endpoint_reports_the_versions_that_produced_the_answers(client):
    body = client.get("/api/health").json()
    assert body["ok"] is True
    assert body["prompt_version"] and body["config_version"]
    assert body["corpus"]["documents"] >= 70


def test_the_capability_matrix_is_servable_without_a_conversation(client):
    """AG-4. Build-time binding is what makes this document exist at all."""
    body = client.get("/api/bots").json()
    matrix = body["capability_matrix"]
    bot05 = {t["name"] for t in matrix["BOT-05"]["tools"]}
    bot02 = {t["name"] for t in matrix["BOT-02"]["tools"]}
    assert "commission_statement" in bot02
    assert "commission_statement" not in bot05
    assert "kb_search_motor" not in bot05


def test_the_tool_manifest_publishes_behaviour_metadata(client):
    tools = {t["name"]: t for t in client.get("/api/tools").json()["tools"]}
    assert tools["policy_get"]["auth"] == "authenticated"
    assert tools["policy_get"]["pii"] is True
    assert tools["commission_statement"]["subject"] == "producer_id"
    assert tools["payment_collect"]["confirm"] is True
    assert tools["kb_search_health"]["effect"] == "read"


def test_the_memory_registry_is_servable(client):
    stores = {s["name"]: s for s in client.get("/api/memory").json()["stores"]}
    assert stores["bot_session"]["crosses_lob"] is False
    assert stores["audit"]["erasure"] == "retained for regulatory record"


# --- routing ---------------------------------------------------------------
def test_an_ambiguous_first_turn_asks_instead_of_guessing(client):
    data = _say(client, STRANGER, "hi")
    assert "?" in _reply(data)
    assert _events(data, "bind") == []          # no bot was even bound


def test_an_entity_routes_and_binds_the_health_bot(client):
    data = _say(client, STRANGER, "what is the waiting period for a "
                                  "pre-existing disease")
    bind = _events(data, "bind")[0]
    assert bind["label"] == "BOT-05"
    assert bind["detail"]["lob"] == "health"
    assert "kb_search_health" in bind["detail"]["tools"]
    assert "kb_search_motor" not in bind["detail"]["tools"]


def test_the_prior_holds_through_an_ambiguous_follow_up(client):
    _say(client, STRANGER, "what is the waiting period for pre-existing disease")
    data = _say(client, STRANGER, "and what about that")
    resolve = [e for e in _events(data, "resolve") if e["label"] == "lob"][0]
    assert resolve["detail"]["value"] == "health"
    assert resolve["detail"]["source"] == "history"


def test_a_mid_conversation_switch_binds_a_different_bot(client):
    """RS-8. And the health session is untouched - suspension is free when
    the sessions are keyed apart."""
    _say(client, STRANGER, "tell me about my health cover waiting period")
    data = _say(client, STRANGER, "actually my car renewal is due")
    assert _events(data, "resolve", "switch")
    assert _events(data, "bind")[0]["label"] == "BOT-06"

    back = _say(client, STRANGER, "back to the health policy please")
    assert _events(back, "bind")[0]["label"] == "BOT-05"


def test_the_entry_link_cannot_promote_a_customer_to_a_producer(client):
    """RS-9 through the API: `?persona=agent` is a request, not an answer."""
    data = _say(client, STRANGER, "show me my commission statement",
                persona="agent", lob="motor")
    assert _events(data, "guardrail", "persona_escalation_refused")
    bind = _events(data, "bind")[0]
    assert bind["detail"]["persona"] == "customer"
    assert "commission_statement" not in bind["detail"]["tools"]


def test_a_producer_is_recognised_from_the_directory(client):
    data = _say(client, PRODUCER, "what is my commission this month",
                lob="motor")
    bind = _events(data, "bind")[0]
    assert bind["label"] == "BOT-02"
    assert "commission_statement" in bind["detail"]["tools"]


def test_holdings_route_a_customer_who_only_holds_one_line(client):
    data = _say(client, HEALTH_ONLY, "hello")
    assert _events(data, "bind")[0]["detail"]["lob"] == "health"


# --- retrieval and citations ----------------------------------------------
def test_an_answer_carries_a_citation_and_the_trace_carries_the_rejects(client):
    data = _say(client, STRANGER, "what is the waiting period for a "
                                  "pre-existing disease")
    retrieve = _events(data, "retrieve")[0]["detail"]
    assert retrieve["accepted"] >= 1
    assert retrieve["rejected"] >= 1                       # OB-4
    assert retrieve["rejected_chunks"][0]["rejected"]
    assert all("score" in c for c in retrieve["accepted_chunks"])

    guard = _events(data, "guardrail", "citations")[0]["detail"]
    assert guard["cited"] and guard["refused"] is False

    # The reply cites NUMBERS and the trace names the documents behind them.
    # This used to assert the chunk id appeared in the reply, and passed only
    # because a regex found a stray digit inside `PHS-POLICY_WORDING-V2#1`.
    from app.agents.citations import refs_in

    assert refs_in(_reply(data)), "no numbered citation in the reply"
    assert guard["invented_refs"] == []


def test_a_question_with_no_approved_source_refuses_and_offers_a_human(client):
    """KB-5. What matters is the outcome, not which layer produced it: the
    retrieval returns nothing, the answer offers a human, and nothing in the
    reply is a claim about cover."""
    data = _say(client, STRANGER, "does my health plan cover a holiday in "
                                  "Lisbon and a new laptop")
    assert _events(data, "retrieve")[0]["detail"]["accepted"] == 0
    guard = _events(data, "guardrail", "citations")
    assert guard and guard[0]["detail"]["grounded_by"] == "nothing"
    assert guard[0]["detail"]["cited"] == []
    assert "colleague" in _reply(data)


def test_an_answer_with_no_tool_behind_it_is_not_let_through(client):
    """The hole KB-4 would otherwise leave: a model that skips retrieval and
    answers from general knowledge. Checking only the turns that retrieved
    lets exactly that answer through."""
    from app.agents import citations

    out, report = citations.enforce(
        "Health insurance does not usually cover holidays or electronics.",
        [], other_tool_evidence=False)
    assert report["refused"] is True
    assert "colleague" in out


def test_retrieval_never_crosses_the_line_of_business(client):
    data = _say(client, STRANGER, "my car insurance no claim bonus")
    retrieve = _events(data, "retrieve")[0]["detail"]
    assert retrieve["dropped_by_prefilter"]["lob"] > 0
    for c in retrieve["accepted_chunks"]:
        assert not c["chunk_id"].startswith("PHS")


# --- authorization ---------------------------------------------------------
def test_a_customer_asking_for_commission_gets_no_such_tool(client):
    """The tool is not bound, so it cannot even be named."""
    data = _say(client, STRANGER, "what commission do you pay on motor",
                lob="motor")
    assert "commission_statement" not in _events(data, "bind")[0]["detail"]["tools"]
    assert not _events(data, "tool", "commission_statement")


def test_a_write_tool_is_refused_without_consent(client):
    """AG-5: the server check runs whatever the client bound, and a consent
    refusal is a refusal the CUSTOMER can lift.

    This used to assert on a `confirmation_required` gate - the parking
    mechanism that intercepted the call and wrote the question into the
    conversation itself. Asking is the model's job now, so what is left to
    check here is the part that is not the model's job: a write with no
    consent behind it does not run.
    """
    from app.agents.graph import agent_for
    from app.agents.registry import BOT_05
    from app.obs.trace import Trace

    out = agent_for(BOT_05)._run_tool(
        "quote_create_health",
        {"product_id": "PHS", "sum_insured": 500000,
         "member_ages": [40], "city": "Pune"},
        {"persona": "customer", "lob": "health", "authenticated": True,
         "customer_id": "C-10001", "user_id": CUSTOMER, "consent": {}},
        Trace())
    assert out["error"] == "consent_required"
    assert out["purpose"] == "quotation"
    assert "consent_grant" in out["try_instead"]

# --- confirmation and idempotency -----------------------------------------
def test_the_audit_record_spans_every_line_of_business(client):
    """OB-3. Separation governs runtime context, not the record."""
    _say(client, CUSTOMER, "what is my health waiting period")
    _say(client, CUSTOMER, "and my car policy no claim bonus")
    entries = client.get(f"/api/audit/{CUSTOMER}").json()["entries"]
    lobs = {e["lob"] for e in entries if e["lob"]}
    assert {"health", "motor"} <= lobs
    for e in entries:
        assert e["prompt_version"] and e["config_version"] and e["trace_id"]


def test_erasure_reports_every_store(client):
    _say(client, CUSTOMER, "what is my health waiting period")
    report = client.post(f"/api/erase/{CUSTOMER}").json()["report"]
    assert report["audit"] == "retained"
    assert report["bot_session"] >= 1


def test_a_step_up_is_a_separate_act_from_the_conversation(client):
    before = client.get(f"/api/session/{PRODUCER}").json()["session"]
    assert before["persona_stepup_at"] is None
    client.post("/api/stepup", json={"user_id": PRODUCER})
    after = client.get(f"/api/session/{PRODUCER}").json()["session"]
    assert after["persona_stepup_at"] and after["profile"]["authenticated"]


# --- channel ---------------------------------------------------------------
def test_the_whatsapp_webhook_parses_without_the_rest_being_wired(client):
    """CH-4: statuses[] are delivery receipts for messages WE sent. Counting
    them as inbound is the classic first-week infinite loop."""
    payload = {"entry": [{"changes": [{"value": {
        "metadata": {"phone_number_id": "PN1"},
        "contacts": [{"wa_id": "919820000009",
                      "profile": {"name": "Priya"}}],
        "messages": [{"id": "wamid.1", "from": "919820000009",
                      "timestamp": "1770000000", "type": "text",
                      "text": {"body": "hello"}}]}}]}]}
    assert client.post("/webhook/whatsapp", json=payload).json()["parsed"] == 1

    receipts = {"entry": [{"changes": [{"value": {
        "metadata": {"phone_number_id": "PN1"},
        "statuses": [{"id": "wamid.1", "status": "delivered"}]}}]}]}
    assert client.post("/webhook/whatsapp", json=receipts).json()["parsed"] == 0


def test_corpus_scope_is_injected_not_offered_to_the_model(client):
    """KB-2. A scope the model can pass is a scope the model can CHOOSE.

    Left as an argument, two things broke at once: the agent bot never
    reached its own agent-scope corpus because the model did not know to ask
    for it, and a customer bot's model could have asked for it and been
    handed the commission grid by a pre-filter doing exactly what it was
    told.
    """
    tools = {t["name"]: t for t in client.get("/api/tools").json()["tools"]}
    for name in ("kb_search_health", "kb_search_motor"):
        props = tools[name]["input_schema"]["properties"]
        assert "scopes" not in props and "_scopes" not in props
        assert set(props) == {"query", "as_of"}


def test_the_agent_bot_actually_reaches_agent_scope_content(client):
    """The other half: BOT-02 gets the agent corpus without asking for it."""
    from app.mcpserver.tools.knowledge import kb_search_motor

    customer = kb_search_motor("commission grid own damage rates",
                               _scopes=["public"])
    agent = kb_search_motor("commission grid own damage rates",
                            _scopes=["public", "agent"])
    assert not any(c["chunk_id"].startswith("M-COMM-GRID")
                   for c in customer["chunks"])
    assert any(c["chunk_id"].startswith("M-COMM-GRID")
               for c in agent["chunks"])


def test_a_customer_turn_searches_only_the_public_corpus(client):
    data = _say(client, STRANGER, "what is the no claim bonus on my car")
    retrieve = _events(data, "retrieve")
    assert retrieve
    assert _events(data, "bind")[0]["detail"]["corpus_scope"] == ["public"]
    for e in retrieve:
        for c in e["detail"]["accepted_chunks"] + e["detail"]["rejected_chunks"]:
            assert not c["chunk_id"].startswith("M-COMM-GRID")


