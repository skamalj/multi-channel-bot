"""Issuance: the application and the gate chain (CO-5), and claims.

The gate chain is the reason these are separate, narrow tools rather than one
`issue_policy(payload)`. KYC, underwriting or inspection, and payment each
have their own decision, their own pending state and their own audit line.
`policy_issue` refuses while any blocking gate is unclear and NAMES the gate,
because "your policy is issued" said over a pending gate is the single most
expensive sentence in this domain.

Nothing here decides eligibility. `underwriting_decision` asks the core for a
decision made by rules in products.yaml and hands the answer back; a
declinature is communicated by an underwriter, never in a chat thread.
"""
from __future__ import annotations

from app.coremock import store
from app.coremock.store import CoreError
from app.mcpserver.registry import tool


@tool(tags={"lob": "health|motor", "persona": "customer|agent"},
      effect="write", authority="core", pii=True, auth="identified",
      subject="application_id")
def application_start(quote_id: str, _customer_id: str | None = None,
                      _user_id: str | None = None,
                      _idempotency_key: str | None = None) -> dict:
    """Open an application from a rated quote and initialise its gate chain.

    Refuses on an expired quote: the quote is re-rated first, never revived."""
    try:
        return store.application_start(quote_id, _customer_id or "unknown",
                                       user_id=_user_id or "",
                                       idempotency_key=_idempotency_key)
    except CoreError as exc:
        return exc.as_result()


@tool(tags={"lob": "health|motor", "persona": "customer|agent"},
      authority="core", auth="identified", subject="application_id")
def application_status(application_id: str) -> dict:
    """Where an application has got to, gate by gate. Read this before saying
    anything to the customer about whether they are covered."""
    try:
        return store.application_get(application_id)
    except CoreError as exc:
        return exc.as_result()


@tool(tags={"lob": "health|motor", "persona": "customer|agent"},
      effect="write", authority="core", pii=True, auth="identified",
      consent_purpose="kyc", subject="application_id")
def kyc_submit(application_id: str, document_type: str,
               reference: str, _idempotency_key: str | None = None) -> dict:
    """Submit a KYC document reference. document_type is pan,
    aadhaar_offline_xml, ckyc or passport.

    Submit a REFERENCE, never a full identifier: do not ask the customer to
    type a full Aadhaar number, a card number or a one-time password into a
    chat, and never repeat one back."""
    try:
        return store.kyc_verify(application_id, document_type, reference)
    except CoreError as exc:
        return exc.as_result()


@tool(tags={"lob": "health", "persona": "customer|agent"},
      effect="write", authority="core", pii=True, auth="identified",
      subject="application_id")
def underwriting_decision(application_id: str,
                          declared_conditions: list[str] | None = None,
                          oldest_member_age: int | None = None,
                          _idempotency_key: str | None = None) -> dict:
    """Ask underwriting for a decision. Returns cleared, referred or declined.

    The rules live in configuration; the model never decides eligibility and
    never communicates a declinature."""
    try:
        return store.underwrite(application_id, declared_conditions,
                                oldest_member_age)
    except CoreError as exc:
        return exc.as_result()


@tool(tags={"lob": "motor", "persona": "customer|agent"},
      effect="write", authority="core", auth="identified",
      subject="application_id")
def inspection_schedule(application_id: str, slot: str,
                        _idempotency_key: str | None = None) -> dict:
    """Book a pre-inspection slot. Cover starts from a clean report, not from
    the payment."""
    try:
        return store.inspection_schedule(application_id, slot)
    except CoreError as exc:
        return exc.as_result()


@tool(tags={"lob": "motor", "persona": "agent"},
      effect="write", authority="core", auth="authenticated",
      subject="application_id")
def inspection_result(application_id: str, outcome: str, note: str = "",
                      _idempotency_key: str | None = None) -> dict:
    """Record the surveyor's report: clean, damage_noted or declined.

    PRODUCER ONLY - a customer cannot clear their own pre-inspection, and
    cover starts from a clean report rather than from the payment."""
    try:
        return store.inspection_result(application_id, outcome, note)
    except CoreError as exc:
        return exc.as_result()


@tool(tags={"lob": "health|motor", "persona": "customer|agent"},
      effect="write", authority="core", pii=True, auth="authenticated",
      consent_purpose="payment", subject="application_id")
def payment_collect(application_id: str, amount: float, mode: str,
                    _idempotency_key: str | None = None) -> dict:
    """Record premium collection. mode is upi, netbanking, card or cheque.

    Never ask for or accept card numbers, CVV, UPI PIN or an OTP in the
    conversation - this records that a payment was made through the payment
    interface, it does not take payment."""
    try:
        return store.payment_collect(application_id, amount, mode,
                                     idempotency_key=_idempotency_key)
    except CoreError as exc:
        return exc.as_result()


@tool(tags={"lob": "health|motor", "persona": "customer|agent"},
      effect="write", authority="core", pii=True, auth="authenticated",
      subject="application_id")
def policy_issue(application_id: str,
                 _idempotency_key: str | None = None) -> dict:
    """Issue the policy. Refused, with the blocking gate named, until every
    blocking gate has cleared."""
    try:
        return store.policy_issue(application_id,
                                  idempotency_key=_idempotency_key)
    except CoreError as exc:
        return exc.as_result()


# --- claims ---------------------------------------------------------------
@tool(tags={"lob": "health|motor", "persona": "customer|agent"},
      effect="write", authority="core", pii=True, auth="authenticated",
      subject="policy_id")
def claim_register(policy_id: str, lob: str, claim_type: str,
                   incident_date: str, description: str,
                   _idempotency_key: str | None = None) -> dict:
    """Register a claim. Always returns a PENDING status with the documents
    still required - registration opens a file, it does not admit a claim,
    and no amount is committed here."""
    try:
        return store.claim_register(policy_id, lob, claim_type, incident_date,
                                    description,
                                    idempotency_key=_idempotency_key)
    except CoreError as exc:
        return exc.as_result()


@tool(tags={"lob": "health|motor", "persona": "customer|agent"},
      authority="core", pii=True, auth="authenticated")
def claim_status(claim_id: str) -> dict:
    """Status of a registered claim, including what is still outstanding."""
    try:
        return store.claim_status(claim_id)
    except CoreError as exc:
        return exc.as_result()


@tool(tags={"lob": "health", "persona": "customer|agent"},
      effect="write", authority="core", pii=True, auth="authenticated",
      subject="policy_id")
def cashless_preauth(policy_id: str, hospital: str, estimate: float,
                     _idempotency_key: str | None = None) -> dict:
    """Raise a cashless pre-authorisation. The outcome is PENDING by design:
    the approved amount is issued by the TPA after the hospital confirms, and
    no amount may be quoted to the customer before then."""
    try:
        return store.cashless_preauth(policy_id, hospital, estimate,
                                      idempotency_key=_idempotency_key)
    except CoreError as exc:
        return exc.as_result()
