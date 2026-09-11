"""Tags are a build-time filter (`tools_for`); the authorization hook
(`controls._authorize`) is the control that does not depend on it. The
model-visible schema and the behaviour policy both come off the framework
`@tool` object - the schema from `convert_to_openai_tool`, the policy from
`extras`.
"""
from langchain_core.utils.function_calling import convert_to_openai_tool

from app.agents.context import RequestContext
from app.agents.controls import _authorize
from app.agents.registry import BOT_02, BOT_05, BOT_06, capability_matrix
from app.mcpserver.registry import get_tool, tools_for


def _names(spec) -> set[str]:
    return {t.name for t in tools_for(spec.tool_tags)}


def _authz(tool_name: str, ctx: dict, args: dict | None = None):
    """Authorize as the hook does: the tool's `extras` policy against a
    per-call `RequestContext`."""
    tool = get_tool(tool_name)
    return _authorize(tool.extras or {}, RequestContext(**ctx), args or {})


def _schema(tool_name: str) -> dict:
    return convert_to_openai_tool(get_tool(tool_name))["function"]["parameters"]


# -- binding is a build-time tag filter -------------------------------------
def test_bot05_never_gets_agent_only_or_motor_tools():
    names = _names(BOT_05)
    assert "commission_statement" not in names
    assert "kb_search_motor" not in names
    assert "network_garages" not in names
    assert "kb_search_health" in names


def test_bot02_gets_agent_tools_and_no_health_tools():
    names = _names(BOT_02)
    assert "commission_statement" in names
    assert "kb_search_motor" in names
    assert "kb_search_health" not in names
    assert "member_waiting_periods" not in names


def test_the_customer_motor_bot_has_no_commercial_tools():
    names = _names(BOT_06)
    assert "commission_statement" not in names
    assert "kb_search_motor" in names


def test_capability_matrix_is_static_and_printable():
    m = capability_matrix()
    assert set(m) == {"BOT-05", "BOT-02", "BOT-06"}
    for info in m.values():
        assert info["tools"]
        for t in info["tools"]:
            assert {"name", "effect", "authority", "auth", "subject"} <= set(t)


# -- authorization is the control, per call ---------------------------------
def test_server_check_refuses_even_when_the_tool_was_bound():
    """A model can name a tool it was never offered - the tag filter is
    context economy, the hook is the control."""
    ok, why = _authz("commission_statement",
                     {"persona": "customer", "user_id": "x",
                      "authenticated": True, "lob": "motor"})
    assert not ok and "persona" in why


def test_a_producer_cannot_read_another_producers_book():
    """AG-5. Having the tool bound is not permission to read that subject."""
    ctx = {"persona": "agent", "user_id": "919820000001", "lob": "motor",
           "authenticated": True, "producer_id": "P-2201"}
    ok, _ = _authz("commission_statement", ctx,
                   {"producer_id": "P-2201", "period": "2026-08"})
    assert ok
    ok, why = _authz("commission_statement", ctx,
                     {"producer_id": "P-9999", "period": "2026-08"})
    assert not ok and "does not match" in why


def test_a_customer_cannot_read_another_customers_policy():
    ok, why = _authz("policy_get",
                     {"persona": "customer", "user_id": "u",
                      "authenticated": True, "lob": "health",
                      "customer_id": "C-10002"},
                     {"policy_id": "PHS-4471902", "lob": "health"})
    assert not ok and "does not belong" in why


def test_authenticated_tools_refuse_anonymous_callers():
    ok, why = _authz("policy_get",
                     {"persona": "customer", "user_id": "x",
                      "lob": "health", "authenticated": False})
    assert not ok and "authentication" in why


def test_consent_gated_tools_refuse_without_a_current_consent():
    base = {"persona": "customer", "user_id": "x", "lob": "health",
            "authenticated": True}
    ok, why = _authz("quote_create_health", base, {})
    assert not ok and "consent" in why
    ok, _ = _authz("quote_create_health",
                   {**base, "consent": {"quotation": True}}, {})
    assert ok


def test_a_tool_from_the_other_compartment_is_refused_at_call_time():
    """The lob tag selects at build time AND is re-checked per call."""
    ok, why = _authz("kb_search_health",
                     {"persona": "customer", "user_id": "x", "lob": "motor"},
                     {"query": "anything"})
    assert not ok and "out of scope" in why


def test_an_application_id_is_not_permission_to_act_on_it():
    """AG-5, same class of hole as the producer id: an application id in a
    prompt is not permission to pay somebody else's premium."""
    from app.coremock import rating, store

    store.reset_chaos()
    q = store.save_quote(rating.rate_health("PHS", 500000, [34], "Pune", []))
    app = store.application_start(q["quote_id"], "C-10001",
                                  user_id="919820000009")
    args = {"application_id": app["application_id"], "amount": 1.0,
            "mode": "upi"}
    mine = {"persona": "customer", "user_id": "919820000009", "lob": "health",
            "authenticated": True, "consent": {"payment": True}}
    ok, _ = _authz("payment_collect", mine, args)
    assert ok
    theirs = {**mine, "user_id": "919899999999"}
    ok, why = _authz("payment_collect", theirs, args)
    assert not ok and "does not belong" in why
    # A producer servicing their book is a different question, and allowed.
    ok, _ = _authz("payment_collect", {**theirs, "persona": "agent"}, args)
    assert ok


# -- the schema the model sees ----------------------------------------------
def test_schemas_resolve_real_types_not_strings():
    """A schema that types every field as a string sends "1000000" to a
    rating engine. Types come from the real signature, and injected args
    (runtime, entitlements) never appear."""
    props = _schema("quote_create_health")["properties"]
    assert props["sum_insured"]["type"] == "integer"
    assert props["member_ages"]["type"] == "array"
    assert props["member_ages"]["items"]["type"] == "integer"
    assert props["city"]["type"] == "string"
    assert "runtime" not in props and "_idempotency_key" not in props
    assert set(_schema("quote_create_health")["required"]) == {
        "product_id", "sum_insured", "member_ages", "city"}
    # And a description says what a type cannot - that a sum insured is rupees.
    assert "rupees" in props["sum_insured"]["description"].lower()


def test_consent_gates_only_guard_state_changes():
    """A consent purpose only makes sense on a tool that changes something."""
    for t in tools_for():
        e = t.extras or {}
        if e.get("consent_purpose"):
            assert e.get("effect") in ("write", "dispatch"), t.name
