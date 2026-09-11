"""Issuance: the application and the gate chain (CO-5), and claims.

The gate chain is why these are separate, narrow tools and not one
`issue_policy(payload)`. KYC, underwriting or inspection, and payment each have
their own decision, pending state and audit line. `policy_issue` refuses while
any blocking gate is unclear and NAMES the gate, because "your policy is
issued" said over a pending gate is the most expensive sentence in this
domain.

Nothing here decides eligibility. `underwriting_decision` asks the core for a
decision made by rules in configuration and hands the answer back; a
declinature is communicated by an underwriter, never in a chat thread.
"""
from __future__ import annotations

from typing import Annotated, Optional

from langchain.tools import tool, ToolRuntime

from app.agents import confirm
from app.coremock import store
from app.coremock.store import CoreError


@tool(extras={"tags": {"lob": "health|motor", "persona": "customer|agent"},
              "effect": "write", "authority": "core", "pii": True,
              "auth": "identified", "subject": "application_id"})
def application_start(quote_id: str, runtime: ToolRuntime = None) -> dict:
    """Open an application from a rated quote and initialise its gate chain.
    Refuses on an expired quote: the quote is re-rated first, never revived."""
    ctx = getattr(runtime, "context", None)
    key = confirm.token_for("application_start", {"quote_id": quote_id})
    try:
        return store.application_start(
            quote_id, getattr(ctx, "customer_id", None) or "unknown",
            user_id=getattr(ctx, "user_id", None) or "", idempotency_key=key)
    except CoreError as exc:
        return exc.as_result()


@tool(extras={"tags": {"lob": "health|motor", "persona": "customer|agent"},
              "authority": "core", "auth": "identified",
              "subject": "application_id"})
def application_status(application_id: str) -> dict:
    """Where an application has got to, gate by gate. Read this before saying
    anything about whether the customer is covered."""
    try:
        return store.application_get(application_id)
    except CoreError as exc:
        return exc.as_result()


@tool(extras={"tags": {"lob": "health|motor", "persona": "customer|agent"},
              "effect": "write", "authority": "core", "pii": True,
              "auth": "identified", "consent_purpose": "kyc",
              "subject": "application_id"})
def kyc_submit(
    application_id: str,
    document_type: Annotated[str, "pan, aadhaar_offline_xml, ckyc or passport."],
    reference: Annotated[str, "A reference to the document, never a full "
                         "identifier - do not ask for or repeat a full Aadhaar "
                         "or card number."],
) -> dict:
    """Submit a KYC document reference."""
    key = confirm.token_for("kyc_submit", {
        "application_id": application_id, "document_type": document_type,
        "reference": reference})
    try:
        return store.kyc_verify(application_id, document_type, reference)
    except CoreError as exc:
        return exc.as_result()


@tool(extras={"tags": {"lob": "health", "persona": "customer|agent"},
              "effect": "write", "authority": "core", "pii": True,
              "auth": "identified", "subject": "application_id"})
def underwriting_decision(
    application_id: str,
    declared_conditions: Annotated[Optional[list[str]], "Declared health conditions, if any."] = None,
    oldest_member_age: Optional[int] = None,
) -> dict:
    """Ask underwriting for a decision. Returns cleared, referred or declined;
    the model never decides eligibility and never communicates a declinature."""
    key = confirm.token_for("underwriting_decision", {
        "application_id": application_id,
        "declared_conditions": declared_conditions,
        "oldest_member_age": oldest_member_age})
    try:
        return store.underwrite(application_id, declared_conditions,
                                oldest_member_age)
    except CoreError as exc:
        return exc.as_result()


@tool(extras={"tags": {"lob": "motor", "persona": "customer|agent"},
              "effect": "write", "authority": "core", "auth": "identified",
              "subject": "application_id"})
def inspection_schedule(
    application_id: str,
    slot: Annotated[str, "The inspection slot to book."],
) -> dict:
    """Book a pre-inspection slot. Cover starts from a clean report, not from
    the payment."""
    key = confirm.token_for("inspection_schedule", {
        "application_id": application_id, "slot": slot})
    try:
        return store.inspection_schedule(application_id, slot)
    except CoreError as exc:
        return exc.as_result()


@tool(extras={"tags": {"lob": "motor", "persona": "agent"},
              "effect": "write", "authority": "core", "auth": "authenticated",
              "subject": "application_id"})
def inspection_result(
    application_id: str,
    outcome: Annotated[str, "clean, damage_noted or declined."],
    note: str = "",
) -> dict:
    """Record the surveyor's report. Producer only - a customer cannot clear
    their own pre-inspection, and cover starts from a clean report."""
    key = confirm.token_for("inspection_result", {
        "application_id": application_id, "outcome": outcome, "note": note})
    try:
        return store.inspection_result(application_id, outcome, note)
    except CoreError as exc:
        return exc.as_result()


@tool(extras={"tags": {"lob": "health|motor", "persona": "customer|agent"},
              "effect": "write", "authority": "core", "pii": True,
              "auth": "authenticated", "consent_purpose": "payment",
              "subject": "application_id"})
def payment_collect(
    application_id: str,
    amount: Annotated[float, "Premium amount in rupees."],
    mode: Annotated[str, "upi, netbanking, card or cheque."],
) -> dict:
    """Record premium collection. This records that a payment was made through
    the payment interface; it does not take payment. Never ask for or accept a
    card number, CVV, UPI PIN or OTP in the conversation."""
    key = confirm.token_for("payment_collect", {
        "application_id": application_id, "amount": amount, "mode": mode})
    try:
        return store.payment_collect(application_id, amount, mode,
                                     idempotency_key=key)
    except CoreError as exc:
        return exc.as_result()


@tool(extras={"tags": {"lob": "health|motor", "persona": "customer|agent"},
              "effect": "write", "authority": "core", "pii": True,
              "auth": "authenticated", "subject": "application_id"})
def policy_issue(application_id: str) -> dict:
    """Issue the policy. Refused, with the blocking gate named, until every
    blocking gate has cleared."""
    key = confirm.token_for("policy_issue", {"application_id": application_id})
    try:
        return store.policy_issue(application_id, idempotency_key=key)
    except CoreError as exc:
        return exc.as_result()


# --- claims ---------------------------------------------------------------
@tool(extras={"tags": {"lob": "health|motor", "persona": "customer|agent"},
              "effect": "write", "authority": "core", "pii": True,
              "auth": "authenticated", "subject": "policy_id"})
def claim_register(
    policy_id: str,
    lob: Annotated[str, "health or motor."],
    claim_type: str,
    incident_date: Annotated[str, "YYYY-MM-DD."],
    description: str,
) -> dict:
    """Register a claim. Always returns PENDING with the documents still
    required - registration opens a file, it does not admit a claim, and no
    amount is committed here."""
    key = confirm.token_for("claim_register", {
        "policy_id": policy_id, "lob": lob, "claim_type": claim_type,
        "incident_date": incident_date, "description": description})
    try:
        return store.claim_register(policy_id, lob, claim_type, incident_date,
                                    description, idempotency_key=key)
    except CoreError as exc:
        return exc.as_result()


@tool(extras={"tags": {"lob": "health|motor", "persona": "customer|agent"},
              "authority": "core", "pii": True, "auth": "authenticated"})
def claim_status(claim_id: str) -> dict:
    """Status of a registered claim, including what is still outstanding."""
    try:
        return store.claim_status(claim_id)
    except CoreError as exc:
        return exc.as_result()


@tool(extras={"tags": {"lob": "health", "persona": "customer|agent"},
              "effect": "write", "authority": "core", "pii": True,
              "auth": "authenticated", "subject": "policy_id"})
def cashless_preauth(
    policy_id: str,
    hospital: str,
    estimate: Annotated[float, "Estimated amount in rupees."],
) -> dict:
    """Raise a cashless pre-authorisation. PENDING by design: the approved
    amount is issued by the TPA after the hospital confirms, and no amount may
    be quoted to the customer before then."""
    key = confirm.token_for("cashless_preauth", {
        "policy_id": policy_id, "hospital": hospital, "estimate": estimate})
    try:
        return store.cashless_preauth(policy_id, hospital, estimate,
                                      idempotency_key=key)
    except CoreError as exc:
        return exc.as_result()
