"""Human-in-the-loop for policy issuance (agent-wait, async mode).

The request rides on the tool (@hitl publishes and returns pending); the
decision comes back as a new message and `on_decision` issues it. These prove
the receive side offline against the in-memory ledger: exactly-once, gate-gated,
reject is final, and a crash mid-flight still issues once because the core
replays on its idempotency key.
"""
from langgraph_wait.hitl import question_id_for

from app.agents import approvals
from app.coremock import rating, store


def _cleared_application() -> str:
    """A motor application with every blocking gate cleared, ready to issue."""
    store.reset_chaos()
    q = store.save_quote(rating.rate_motor("PMS", 480000, 35, 5.0, ["ZD"]))
    app = store.application_start(q["quote_id"], "C-10001", user_id="u1")
    aid = app["application_id"]
    store.kyc_verify(aid, "pan", "ABCDE1234F")
    store.inspection_schedule(aid, "tomorrow 10:00")
    store.inspection_result(aid, "clean")
    store.payment_collect(aid, q["gross_premium"], "upi")
    return aid


def _seed_open(thread_id: str, aid: str) -> str:
    """What @hitl(async) publishes: one open row for the issuance question."""
    approvals.reset()
    qid = question_id_for(thread_id, "policy_issue", {"application_id": aid})
    approvals._MEM.put_open({
        "pk": f"THREAD#{thread_id}", "sk": f"WAIT#{qid}", "status": "open",
        "thread_id": thread_id, "question_id": qid,
        "question": {"function": "policy_issue", "args": {"application_id": aid}},
        "allowed_actions": ["approve", "reject"], "expires_at": "never",
        "reply_with": {"thread_id": thread_id, "question_id": qid, "answer": None}})
    return qid


def _msg(thread_id: str, qid: str, action: str) -> dict:
    return {"thread_id": thread_id, "question_id": qid, "answer": {"action": action}}


def test_approval_issues_the_policy_exactly_once():
    aid = _cleared_application()
    qid = _seed_open("t1", aid)

    out = approvals.on_decision(_msg("t1", qid, "approve"))
    assert out["status"] == "executed", out
    assert out["result"]["status"] == "in_force"
    assert "policy" in out["customer"].lower()

    # A duplicate decision (redelivery) is a no-op; the row is already executed.
    again = approvals.on_decision(_msg("t1", qid, "approve"))
    assert again["status"] == "already_executed"
    assert approvals._MEM.get("t1", qid)["status"] == "executed"


def test_rejection_is_final():
    aid = _cleared_application()
    qid = _seed_open("t2", aid)

    out = approvals.on_decision(_msg("t2", qid, "reject"))
    assert out["status"] == "rejected"
    assert "not issued" in out["customer"].lower()
    # A later approval cannot revive a rejected question.
    late = approvals.on_decision(_msg("t2", qid, "approve"))
    assert late["status"] == "already_rejected"


def test_a_crash_after_claim_still_issues_once():
    """Row left 'executing' by a crash before the core call. A retry re-runs
    the core (it replays on its key) and finishes - it does not refuse."""
    aid = _cleared_application()
    qid = _seed_open("t3", aid)
    approvals._MEM.claim("t3", qid)                 # claimed, then "crashed"
    assert approvals._MEM.get("t3", qid)["status"] == "executing"

    out = approvals.on_decision(_msg("t3", qid, "approve"))
    assert out["status"] == "executed" and out["result"]["status"] == "in_force"


def test_a_reject_arriving_after_a_claim_cannot_overturn_it():
    """Every write is conditional: a reject that arrives once the row is
    executing is a no-op, not a row moved backwards to rejected."""
    aid = _cleared_application()
    qid = _seed_open("t6", aid)
    approvals._MEM.claim("t6", qid)                 # row is now executing
    out = approvals.on_decision(_msg("t6", qid, "reject"))
    assert out["status"] == "already_executing"
    assert approvals._MEM.get("t6", qid)["status"] == "executing"


def test_blocked_is_terminal():
    """A gate that regressed after approval blocks the row; a later approval
    cannot revive it - a human resets the row first (runbook)."""
    aid = _cleared_application()
    qid = _seed_open("t7", aid)
    approvals._MEM.claim("t7", qid)
    approvals._MEM.transition("t7", qid, "executing", "blocked", gate="x")
    out = approvals.on_decision(_msg("t7", qid, "approve"))
    assert out["status"] == "already_blocked"


def test_an_unknown_question_is_handled_not_executed():
    approvals.reset()
    out = approvals.on_decision(_msg("t4", "deadbeef" * 4, "approve"))
    assert out["status"] == "unknown_question"


def test_publish_then_decide_is_a_closed_loop():
    """Our announcer wiring writes the open row that the receive side reads:
    publish (what @hitl does internally) -> row -> on_decision -> issued."""
    from agent_wait import Question, publish

    aid = _cleared_application()
    approvals.reset()
    qid = question_id_for("t5", "policy_issue", {"application_id": aid})
    publish([Question(qid, {"function": "policy_issue",
                            "args": {"application_id": aid}},
                      approvals.issuance_policy(),
                      source={"function": "policy_issue"})],
            "t5", approvals.announcers())

    row = approvals._MEM.get("t5", qid)
    assert row and row["status"] == "open"
    out = approvals.on_decision(_msg("t5", qid, "approve"))
    assert out["status"] == "executed" and out["result"]["status"] == "in_force"


def test_issuance_is_not_requested_until_the_gate_chain_is_clear():
    """controls refuses policy_issue - so @hitl never publishes an approval -
    while a blocking gate is unclear; a human is only asked what can execute."""
    from app.agents.controls import _issuance_gates_blocking

    store.reset_chaos()
    q = store.save_quote(rating.rate_motor("PMS", 480000, 35, 5.0, ["ZD"]))
    app = store.application_start(q["quote_id"], "C-10001", user_id="u1")
    aid = app["application_id"]
    store.kyc_verify(aid, "pan", "ABCDE1234F")
    store.inspection_schedule(aid, "tomorrow 10:00")            # pending

    assert _issuance_gates_blocking(aid), "inspection gate should still block"
    store.inspection_result(aid, "clean")
    store.payment_collect(aid, q["gross_premium"], "upi")
    assert _issuance_gates_blocking(aid) == []
