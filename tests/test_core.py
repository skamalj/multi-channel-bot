"""The mock core behaves like a core: gates, pending states, a 429, one
interface that is not there, and writes that survive a retry."""
import pytest

from app.coremock import rating, store
from app.coremock.store import CoreError


def _quote_health():
    q = rating.rate_health("PHS", 500000, [34], "Pune", [])
    return store.save_quote(q)


def test_a_quote_carries_a_validity_and_an_expired_one_is_repriced():
    """CO-3. An expired quote is re-priced, never resurrected."""
    q = _quote_health()
    got = store.get_quote(q["quote_id"])
    assert got["expired"] is False
    assert got["next_step"] == "proceed_to_application"

    store._QUOTES[q["quote_id"]]["valid_until"] = "2020-01-01"
    expired = store.get_quote(q["quote_id"])
    assert expired["expired"] is True and expired["next_step"] == "re-rate"

    app = store.application_start(q["quote_id"], "C-10001")
    assert app["error"] == "quote_expired"


def test_the_same_idempotency_key_returns_the_first_result():
    """CO-1. Money is the one place a duplicate is unforgivable."""
    q = _quote_health()
    app = store.application_start(q["quote_id"], "C-10001")
    first = store.payment_collect(app["application_id"], 8000.0, "upi",
                                  idempotency_key="tok-1")
    again = store.payment_collect(app["application_id"], 8000.0, "upi",
                                  idempotency_key="tok-1")
    assert again["idempotent_replay"] is True
    assert again["receipt_id"] == first["receipt_id"]


def test_the_same_key_with_a_different_payload_is_a_client_bug():
    q = _quote_health()
    app = store.application_start(q["quote_id"], "C-10001")
    store.payment_collect(app["application_id"], 8000.0, "upi",
                          idempotency_key="tok-2")
    with pytest.raises(CoreError) as exc:
        store.payment_collect(app["application_id"], 12000.0, "upi",
                              idempotency_key="tok-2")
    assert exc.value.code == "idempotency_key_reused"


def test_issuance_is_refused_until_every_blocking_gate_clears():
    """CO-5. The refusal NAMES the gate - "your policy is issued" said over a
    pending gate is the most expensive sentence in this domain."""
    q = _quote_health()
    app = store.application_start(q["quote_id"], "C-10001")
    assert [g["id"] for g in app["gates"]] == ["kyc", "uw", "payment"]

    blocked = store.policy_issue(app["application_id"])
    assert blocked["error"] == "gate_not_cleared"
    assert {g["id"] for g in blocked["blocked_by"]} == {"kyc", "uw", "payment"}

    store.kyc_verify(app["application_id"], "pan", "ABCDE1234F")
    store.underwrite(app["application_id"], [], 34)
    still = store.policy_issue(app["application_id"])
    assert still["error"] == "gate_not_cleared"
    assert [g["id"] for g in still["blocked_by"]] == ["payment"]

    store.payment_collect(app["application_id"], q["gross_premium"], "upi")
    issued = store.policy_issue(app["application_id"])
    assert issued["status"] == "in_force" and issued["policy_id"]


def test_a_pending_gate_is_an_outcome_not_a_failure():
    """CO-4. A cheque is not a failed payment; it is a payment that has not
    realised yet, and the difference matters to what the bot says."""
    q = _quote_health()
    app = store.application_start(q["quote_id"], "C-10001")
    res = store.payment_collect(app["application_id"], 1000.0, "cheque")
    assert res["state"] == "pending" and "realisation" in res["detail"]
    assert store.policy_issue(app["application_id"])["error"] == "gate_not_cleared"


def test_kyc_can_land_in_manual_review():
    q = _quote_health()
    app = store.application_start(q["quote_id"], "C-10001")
    res = store.kyc_verify(app["application_id"], "pan", "ABCDE12340")
    assert res["state"] == "pending" and res["sla_hours"] == 24


def test_underwriting_refers_rather_than_declining_on_a_disclosure():
    q = rating.rate_health("PHS", 2000000, [52], "Pune", [])
    saved = store.save_quote(q)
    app = store.application_start(saved["quote_id"], "C-10001")
    res = store.underwrite(app["application_id"], ["diabetes"], 52)
    assert res["state"] == "referred" and res["reasons"]


def test_an_auto_decline_is_not_communicated_by_the_assistant():
    q = _quote_health()
    app = store.application_start(q["quote_id"], "C-10001")
    res = store.underwrite(app["application_id"], ["active_malignancy"], 40)
    assert res["state"] == "declined"
    assert "underwriter" in res["referral_note"]


def test_claim_registration_is_always_pending_and_never_approves():
    claim = store.claim_register("PHS-4471902", "health", "hospitalisation",
                                 "2026-08-01", "planned surgery")
    assert claim["status"] == "registered"
    assert claim["sub_status"] == "pending_documents"
    assert claim["documents_required"]
    assert "approved" not in str(claim).lower()


def test_a_preauth_commits_no_amount():
    res = store.cashless_preauth("PHS-4471902", "Ruby Hall Clinic", 120000)
    assert res["status"] == "pending"
    assert "no amount is committed" in res["note"]


def test_a_claim_cannot_be_registered_against_the_other_line_of_business():
    res = store.claim_register("PHS-4471902", "motor", "own_damage",
                               "2026-08-01", "dent")
    assert res["error"] == "out_of_scope"


def test_the_endorsement_interface_is_honestly_unavailable():
    """CO-4. A typed unavailability beats a bot that claims a change was
    made."""
    import app.config as config_mod

    config_mod.settings.cache_clear()
    with pytest.raises(CoreError) as exc:
        store.endorsement_apply("PHS-4471902", "address", "new address")
    assert exc.value.code == "interface_unavailable"
    assert exc.value.http_status == 503
    assert exc.value.retryable is False


def test_the_quote_interface_throttles_under_burst_and_says_it_is_retryable():
    store.reset_chaos()
    q = rating.rate_health("PHS", 500000, [34], "Pune", [])
    for _ in range(5):
        store.save_quote(dict(q))
    with pytest.raises(CoreError) as exc:
        store.save_quote(dict(q))
    assert exc.value.code == "rate_limited"
    assert exc.value.http_status == 429 and exc.value.retryable is True


def test_a_policy_read_is_scoped_to_the_active_line_of_business():
    """The tool is the third leak path after sessions and retrieval."""
    assert store.get_policy("PHS-4471902", "health")["status"] == "in_force"
    out = store.get_policy("PHS-4471902", "motor")
    assert out["error"] == "out_of_scope"


def test_waiting_periods_are_answered_from_the_policy_date_wording():
    """KB-6 through the core: PHS-4471902 was incepted in 2025."""
    wp = store.waiting_periods("PHS-4471902")
    assert wp["wording_version"] == "V1"
    ped = next(i for i in wp["items"] if i["kind"] == "ped_months")
    assert ped["required_months"] == 48


def test_holdings_are_facts_about_a_line_of_business():
    assert store.holdings_for("919820000009") == {"health": True, "motor": True}
    assert store.holdings_for("919820000002") == {"health": True}
    assert store.holdings_for("nobody") == {}


def test_a_motor_policy_can_actually_be_issued_once_the_surveyor_reports():
    """CO-5's second half. Booking an inspection only makes the gate PENDING;
    without the surveyor's report coming back, a motor policy could never
    issue at all - the chain was missing an operation, not being strict."""
    q = store.save_quote(rating.rate_motor("PMS", 480000, 35, 5.0, ["ZD"]))
    app = store.application_start(q["quote_id"], "C-10001", user_id="u1")
    aid = app["application_id"]

    store.kyc_verify(aid, "pan", "ABCDE1234F")
    store.inspection_schedule(aid, "tomorrow 10:00")
    assert store.policy_issue(aid)["error"] == "gate_not_cleared"

    store.inspection_result(aid, "clean")
    store.payment_collect(aid, q["gross_premium"], "upi")
    issued = store.policy_issue(aid)
    assert issued["status"] == "in_force"
    assert issued["idv"] == 480000


def test_a_customer_cannot_clear_their_own_inspection():
    from app.mcpserver.registry import authorize, get_tool

    spec = get_tool("inspection_result")
    assert spec.tags["persona"] == "agent"
    ok, why = authorize(spec, {"persona": "customer", "user_id": "u",
                               "lob": "motor", "authenticated": True},
                        {"application_id": "A-1", "outcome": "clean"})
    assert not ok and "persona" in why
