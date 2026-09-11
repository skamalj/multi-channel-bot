"""Policy, servicing and agent-only tools.

Two of these are agent-only: `"persona": "agent"` in `extras` keeps them out
of a customer bot's bound set entirely, and the authorization hook re-checks
persona AND subject on every call, so the boundary never depends on the client
filtering correctly. `commission_statement` declares `"subject":
"producer_id"`, which is what turns "may you call this tool" into "may you
call it for THAT producer".
"""
from __future__ import annotations

from typing import Annotated, Optional

from langchain.tools import tool, ToolRuntime

from app.agents import confirm
from app.coremock import store
from app.coremock.store import CoreError


@tool(extras={"tags": {"lob": "health|motor", "persona": "customer|agent"},
              "authority": "core", "pii": True, "auth": "authenticated",
              "subject": "policy_id"})
def policy_get(policy_id: str,
               lob: Annotated[str, "health or motor."]) -> dict:
    """Retrieve one policy: cover, members or vehicle, status and documents."""
    try:
        return store.get_policy(policy_id, lob=lob)
    except CoreError as exc:
        return exc.as_result()


@tool(extras={"tags": {"lob": "health|motor", "persona": "customer|agent"},
              "authority": "core", "pii": True, "auth": "authenticated"})
def policy_list_mine(runtime: ToolRuntime = None) -> list[dict]:
    """List the caller's own policies in the active line of business.

    Takes no arguments: the customer and the line of business come from the
    request context, not the model - a listing tool that accepted a customer
    id from a prompt is the front door into someone else's book."""
    ctx = getattr(runtime, "context", None)
    customer_id = getattr(ctx, "customer_id", None)
    lob = getattr(ctx, "lob", None)
    if not customer_id or not lob:
        return []
    return store.policies_for_customer(customer_id, lob)


@tool(extras={"tags": {"lob": "health", "persona": "customer|agent"},
              "authority": "core", "pii": True, "auth": "authenticated",
              "subject": "policy_id"})
def member_waiting_periods(policy_id: str) -> dict:
    """Waiting-period status for a health policy, answered from the wording in
    force on the policy start date rather than today's."""
    try:
        return store.waiting_periods(policy_id)
    except CoreError as exc:
        return exc.as_result()


@tool(extras={"tags": {"lob": "health", "persona": "customer|agent"},
              "authority": "core"})
def network_hospitals(pincode: str,
                      specialty: Optional[str] = None,
                      limit: int = 5) -> list[dict]:
    """Cashless network hospitals near a PIN code."""
    try:
        return store.network_search("hospital", pincode, specialty, limit)
    except CoreError as exc:
        return [exc.as_result()]


@tool(extras={"tags": {"lob": "motor", "persona": "customer|agent"},
              "authority": "core"})
def network_garages(pincode: str, limit: int = 5) -> list[dict]:
    """Cashless network garages near a PIN code."""
    try:
        return store.network_search("garage", pincode, None, limit)
    except CoreError as exc:
        return [exc.as_result()]


@tool(extras={"tags": {"lob": "motor", "persona": "customer|agent"},
              "authority": "core"})
def vehicle_lookup(registration: str) -> dict:
    """Decode a registration number: make, model, variant, year, fuel, RTO,
    prior insurer and prior no claim bonus."""
    try:
        return store.vehicle_lookup(registration)
    except CoreError as exc:
        return exc.as_result()


@tool(extras={"tags": {"lob": "motor", "persona": "agent"},
              "authority": "core", "pii": True, "auth": "authenticated",
              "subject": "producer_id"})
def commission_statement(
    producer_id: str,
    period: Annotated[str, "YYYY-MM."],
) -> dict:
    """Commission and payout statement for a producer. Agent only; the hook
    refuses when producer_id is not the authenticated producer's own."""
    try:
        return store.commission(producer_id, period)
    except CoreError as exc:
        return exc.as_result()


@tool(extras={"tags": {"lob": "health|motor", "persona": "customer|agent"},
              "effect": "dispatch", "authority": "none"})
def human_handoff(
    reason: Annotated[str, "Why a human is needed, in a few words."],
    summary: Annotated[str, "What has happened so far, for the colleague picking up."],
) -> dict:
    """Transfer to a human with identity, intent, summary and captured fields."""
    return {"status": "queued", "reason": reason, "summary": summary[:500],
            "sla_minutes": 15,
            "note": "a colleague will pick this up with the full context"}


@tool(extras={"tags": {"lob": "health|motor", "persona": "customer|agent"},
              "effect": "write", "authority": "none", "auth": "identified"})
def endorsement_apply(
    policy_id: str,
    endorsement_type: Annotated[str, "What is changing: address, nominee or registration."],
    value: Annotated[str, "The new value."],
) -> dict:
    """Apply a mid-term change to a policy. Not exposed in this environment;
    returns a typed unavailability rather than claiming a change was made."""
    try:
        return store.endorsement_apply(policy_id, endorsement_type, value)
    except CoreError as exc:
        return exc.as_result()


# The purposes a customer can be asked to consent to, and what each is for in
# plain words. A closed set: consent to "everything" is not consent, and a
# purpose the model can invent is a purpose nobody agreed to.
CONSENT_PURPOSES = {
    "quotation": "prepare a quote using the details you have given",
    "kyc": "submit your KYC documents to the insurer",
    "payment": "collect a premium payment from you",
}


@tool(extras={"tags": {"lob": "health|motor", "persona": "customer|agent"},
              "effect": "write", "authority": "none", "auth": "none"})
def consent_grant(
    purpose: Annotated[str, "quotation, kyc or payment - the purpose named in "
                       "the consent_required error being responded to."],
    runtime: ToolRuntime = None,
) -> dict:
    """Record that the customer agreed to something being done for them. Call
    this after a tool was refused for want of consent, and after the customer
    has said yes - their yes is the consent that gets written down."""
    from app.memory.longterm import consent as ledger

    ctx = getattr(runtime, "context", None)
    user_id = getattr(ctx, "user_id", None)
    if purpose not in CONSENT_PURPOSES:
        return {"error": "unknown_purpose", "purpose": purpose,
                "detail": "consent is recorded per purpose, from a fixed set",
                "allowed": sorted(CONSENT_PURPOSES)}
    if not user_id:
        return {"error": "no_subject",
                "detail": "there is nobody to record consent for"}

    # `user_stated` is the truth here: the customer was shown the purpose and
    # said yes. The ledger refuses evidence sourced from a model inference,
    # which is what recording consent nobody gave would be.
    ledger.record(user_id, purpose, True,
                  {"source": "user_stated", "ref": "confirmed in conversation"})

    # Make the grant visible WITHIN this turn. The request context's consent
    # snapshot was read from the ledger at turn start, so without this the
    # model doing the right thing - grant, then retry the gated call in the
    # same turn - would hit the stale snapshot, be refused again, and loop
    # asking for consent it was just given. The next turn rebuilds the context
    # from the ledger, so this only bridges the current one.
    if ctx is not None and isinstance(getattr(ctx, "consent", None), dict):
        ctx.consent[purpose] = True

    return {"status": "recorded", "purpose": purpose,
            "means": CONSENT_PURPOSES[purpose],
            "note": ("consent is recorded for this purpose and the call that "
                     "needed it can now be made")}
