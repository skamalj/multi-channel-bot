"""Policy, servicing and agent-only tools.

Note the two agent-only tools: `persona: agent` keeps them out of BOT-05's
bound set entirely, and `authorize()` re-checks the persona AND the subject
on every call, so the boundary does not depend on the client filtering
correctly. `commission_statement` declares `subject="producer_id"`, which is
what turns "may you call this tool" into "may you call it for THAT producer".
"""
from __future__ import annotations

from app.coremock import store
from app.coremock.store import CoreError
from app.mcpserver.registry import tool


@tool(tags={"lob": "health|motor", "persona": "customer|agent"},
      authority="core", pii=True, auth="authenticated", subject="policy_id")
def policy_get(policy_id: str, lob: str) -> dict:
    """Retrieve one policy: cover, members or vehicle, status and documents.

    Scoped to the active line of business: the tool is the third leak path
    after sessions and retrieval, so it filters by lob itself.
    """
    try:
        return store.get_policy(policy_id, lob=lob)
    except CoreError as exc:
        return exc.as_result()


@tool(tags={"lob": "health|motor", "persona": "customer|agent"},
      authority="core", pii=True, auth="authenticated")
def policy_list_mine(_customer_id: str | None = None,
                     _lob: str | None = None) -> list[dict]:
    """List the caller's own policies in the ACTIVE line of business only.

    The customer id and the line of business are injected from the request
    context, not taken from the model - a listing tool that accepts a
    customer id from a prompt is the front door into somebody else's book."""
    if not _customer_id or not _lob:
        return []
    return store.policies_for_customer(_customer_id, _lob)


@tool(tags={"lob": "health", "persona": "customer|agent"},
      authority="core", pii=True, auth="authenticated", subject="policy_id")
def member_waiting_periods(policy_id: str) -> dict:
    """Waiting-period status for a HEALTH policy, answered from the wording in
    force on the POLICY start date rather than today's wording."""
    try:
        return store.waiting_periods(policy_id)
    except CoreError as exc:
        return exc.as_result()


@tool(tags={"lob": "health", "persona": "customer|agent"}, authority="core")
def network_hospitals(pincode: str, specialty: str | None = None,
                      limit: int = 5) -> list[dict]:
    """Cashless network hospitals near a PIN code."""
    try:
        return store.network_search("hospital", pincode, specialty, limit)
    except CoreError as exc:
        return [exc.as_result()]


@tool(tags={"lob": "motor", "persona": "customer|agent"}, authority="core")
def network_garages(pincode: str, limit: int = 5) -> list[dict]:
    """Cashless network garages near a PIN code."""
    try:
        return store.network_search("garage", pincode, None, limit)
    except CoreError as exc:
        return [exc.as_result()]


@tool(tags={"lob": "motor", "persona": "customer|agent"}, authority="core")
def vehicle_lookup(registration: str) -> dict:
    """Decode a registration number: make, model, variant, year, fuel, RTO,
    prior insurer and prior no claim bonus."""
    try:
        return store.vehicle_lookup(registration)
    except CoreError as exc:
        return exc.as_result()


@tool(tags={"lob": "motor", "persona": "agent"},
      authority="core", pii=True, auth="authenticated", subject="producer_id")
def commission_statement(producer_id: str, period: str) -> dict:
    """Commission and payout statement for a producer, period as YYYY-MM.
    AGENT ONLY.

    `authorize()` additionally refuses when the caller's identity does not
    match producer_id - having the tool bound is not permission to read
    somebody else's book.
    """
    try:
        return store.commission(producer_id, period)
    except CoreError as exc:
        return exc.as_result()


@tool(tags={"lob": "health|motor", "persona": "customer|agent"},
      effect="dispatch", authority="none", confirm=False, idempotent=False)
def human_handoff(reason: str, summary: str) -> dict:
    """Transfer to a human with identity, intent, summary and captured fields.

    Use this only AFTER searching the approved sources or calling the tool
    that would answer the question. Handing off a question you never looked
    up is not caution, it is a bot that does not work.

    Deliberately NOT confirmation-gated: a handoff is the safe direction, and
    making the customer confirm their way out of a bot that is failing them
    is the wrong place to be careful."""
    return {"status": "queued", "reason": reason, "summary": summary[:500],
            "sla_minutes": 15,
            "note": "a colleague will pick this up with the full context"}


@tool(tags={"lob": "health|motor", "persona": "customer|agent"},
      effect="write", authority="none", auth="identified")
def endorsement_apply(policy_id: str, endorsement_type: str, value: str,
                      _idempotency_key: str | None = None) -> dict:
    """Apply a mid-term change to a policy: address, nominee, registration.

    This interface is not exposed in this environment and returns a typed
    unavailability - which is a truthful answer, and better than a bot that
    claims a change was made."""
    try:
        return store.endorsement_apply(policy_id, endorsement_type, value)
    except CoreError as exc:
        return exc.as_result()
