"""Tags are a build-time filter, and the capability matrix is the artefact
that proves it. `authorize()` is the control that does not depend on it."""
from app.agents.registry import BOT_02, BOT_05, BOT_06, capability_matrix
from app.mcpserver.registry import (authorize, get_tool, json_schema,
                                    list_tools)


def test_bot05_never_gets_agent_only_or_motor_tools():
    names = {t.name for t in list_tools(match=BOT_05.tool_tags)}
    assert "commission_statement" not in names
    assert "kb_search_motor" not in names
    assert "network_garages" not in names
    assert "kb_search_health" in names


def test_bot02_gets_agent_tools_and_no_health_tools():
    names = {t.name for t in list_tools(match=BOT_02.tool_tags)}
    assert "commission_statement" in names
    assert "kb_search_motor" in names
    assert "kb_search_health" not in names
    assert "member_waiting_periods" not in names


def test_the_customer_motor_bot_has_no_commercial_tools():
    names = {t.name for t in list_tools(match=BOT_06.tool_tags)}
    assert "commission_statement" not in names
    assert "kb_search_motor" in names


def test_capability_matrix_is_static_and_printable():
    m = capability_matrix()
    assert set(m) == {"BOT-05", "BOT-02", "BOT-06"}
    for info in m.values():
        assert info["tools"]
        for t in info["tools"]:
            assert {"name", "effect", "authority", "auth", "confirm"} <= set(t)


def test_server_check_refuses_even_when_the_tool_was_bound():
    """A model can name a tool it was never offered - the client filter is
    context economy, the server check is the control."""
    spec = get_tool("commission_statement")
    ok, why = authorize(spec, {"persona": "customer", "user_id": "x",
                               "authenticated": True, "lob": "motor"})
    assert not ok and "persona" in why


def test_a_producer_cannot_read_another_producers_book():
    """AG-5. Having the tool bound is not permission to read that subject."""
    spec = get_tool("commission_statement")
    ctx = {"persona": "agent", "user_id": "919820000001", "lob": "motor",
           "authenticated": True, "producer_id": "P-2201"}
    ok, _ = authorize(spec, ctx, {"producer_id": "P-2201", "period": "2026-08"})
    assert ok
    ok, why = authorize(spec, ctx, {"producer_id": "P-9999",
                                    "period": "2026-08"})
    assert not ok and "another producer" in why


def test_a_customer_cannot_read_another_customers_policy():
    spec = get_tool("policy_get")
    ctx = {"persona": "customer", "user_id": "u", "authenticated": True,
           "lob": "health", "customer_id": "C-10002"}
    ok, why = authorize(spec, ctx, {"policy_id": "PHS-4471902",
                                    "lob": "health"})
    assert not ok and "does not belong" in why


def test_authenticated_tools_refuse_anonymous_callers():
    spec = get_tool("policy_get")
    ok, why = authorize(spec, {"persona": "customer", "user_id": "x",
                               "lob": "health", "authenticated": False})
    assert not ok and "authentication" in why


def test_consent_gated_tools_refuse_without_a_current_consent():
    spec = get_tool("quote_create_health")
    base = {"persona": "customer", "user_id": "x", "lob": "health",
            "authenticated": True}
    ok, why = authorize(spec, base, {})
    assert not ok and "consent" in why
    ok, _ = authorize(spec, base | {"consent": {"quotation": True}}, {})
    assert ok


def test_a_tool_from_the_other_compartment_is_refused_at_call_time():
    """The lob tag selects at build time AND is re-checked per call."""
    spec = get_tool("kb_search_health")
    ok, why = authorize(spec, {"persona": "customer", "user_id": "x",
                               "lob": "motor"}, {"query": "anything"})
    assert not ok and "out of scope" in why


def test_mutating_tools_declare_confirmation_and_idempotency():
    """AG-6 is a property of the manifest, not of one call site."""
    for t in list_tools():
        if t.effect in ("write", "dispatch") and t.name != "human_handoff":
            assert t.confirm, f"{t.name} mutates without confirmation"
            assert t.idempotent, f"{t.name} mutates without an idempotency key"


def test_schemas_resolve_real_types_not_strings():
    """`from __future__ import annotations` makes every annotation a string;
    a schema that types every field as a string sends "1000000" to a rating
    engine."""
    schema = json_schema(get_tool("quote_create_health"))
    props = schema["properties"]
    assert props["sum_insured"]["type"] == "integer"
    assert props["member_ages"]["type"] == "array"
    assert props["member_ages"]["items"] == {"type": "integer"}
    assert props["city"]["type"] == "string"
    assert "_idempotency_key" not in props          # injected, never modelled
    assert set(schema["required"]) == {"product_id", "sum_insured",
                                       "member_ages", "city"}

    # And each one says what it IS, which a type cannot: the signature
    # cannot tell a model that a sum insured is in rupees rather than lakhs.
    assert "rupees" in props["sum_insured"]["description"].lower()
    assert props["city"]["description"].endswith("Required.")
    assert props["addons"]["description"].endswith("Optional.")


def test_closed_sets_are_in_the_schema_not_in_the_models_memory():
    """AG-9. A product id the model has to recall is one it will invent, and
    the customer is then asked to confirm a call that cannot succeed."""
    schema = json_schema(get_tool("quote_create_motor"))
    assert schema["properties"]["product_id"]["enum"] == ["PMS", "PTS"]
    # On a list parameter the closed set constrains the ELEMENTS.
    addons = schema["properties"]["addons"]
    assert addons["type"] == "array"
    assert "enum" not in addons
    assert set(addons["items"]["enum"]) == {"ZD", "RSA", "EP", "CS"}

    health = json_schema(get_tool("quote_create_health"))
    assert set(health["properties"]["product_id"]["enum"]) == {"PHS", "PHST",
                                                               "PHSR"}


def test_an_application_id_is_not_permission_to_act_on_it():
    """AG-5, same class of hole as the producer id.

    An application id in a prompt is not permission to clear somebody else's
    KYC gate or pay their premium. Checking only that *an* identity exists
    lets any authenticated caller drive another customer's issuance.
    """
    from app.coremock import rating, store

    store.reset_chaos()
    q = store.save_quote(rating.rate_health("PHS", 500000, [34], "Pune", []))
    app = store.application_start(q["quote_id"], "C-10001",
                                  user_id="919820000009")
    spec = get_tool("payment_collect")
    args = {"application_id": app["application_id"], "amount": 1.0,
            "mode": "upi"}

    mine = {"persona": "customer", "user_id": "919820000009", "lob": "health",
            "authenticated": True, "consent": {"payment": True}}
    ok, _ = authorize(spec, mine, args)
    assert ok

    theirs = mine | {"user_id": "919899999999"}
    ok, why = authorize(spec, theirs, args)
    assert not ok and "does not belong" in why

    # A producer servicing their book is a different question, and allowed.
    ok, _ = authorize(spec, theirs | {"persona": "agent"}, args)
    assert ok
