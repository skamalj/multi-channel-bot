"""Confirmation and idempotency for mutating tools (AG-6).

There is no interrupt here, and there could not be. On Lambda the process
ends when the reply is sent and hours pass before the next message, so
"waiting for a confirmation" is **a field that is still empty**, not a parked
node. The pending call lives in the bot session and the next turn either
finds an affirmative answer or it does not.

The idempotency key is the confirmation token, deliberately. Keying on the
message id would give the confirming turn a different key from the proposing
turn, which is precisely the retry that must not double-charge. The token
identifies the OPERATION; it survives the round trip; a replay of the same
token returns the first result rather than performing the write again.
"""
from __future__ import annotations

import hashlib
import json
import re
import time
from typing import Any

AFFIRMATIVE = re.compile(
    r"^\s*(y|ya|yes|yep|yeah|ok|okay|sure|please do|go ahead|do it|confirm|"
    r"confirmed|proceed|haan|haan ji|ji|theek hai|thik hai|correct|that's "
    r"right|thats right)\b", re.I)

NEGATIVE = re.compile(
    r"^\s*(n|no|nope|not now|don'?t|do not|cancel|stop|wait|hold on|nahi|"
    r"nahin|later|never mind|nevermind)\b", re.I)

TTL_S = 30 * 60


def token_for(tool_name: str, args: dict[str, Any]) -> str:
    """Stable across the round trip: same tool, same arguments, same token."""
    body = json.dumps({"t": tool_name, "a": args}, sort_keys=True, default=str)
    return hashlib.sha256(body.encode("utf-8")).hexdigest()[:16]


_LABELS = {
    "quote_create_health": "create a health quote",
    "quote_create_motor": "create a motor quote",
    "application_start": "open an application",
    "kyc_submit": "submit a KYC reference",
    "underwriting_decision": "send this to underwriting",
    "inspection_schedule": "book a pre-inspection",
    "payment_collect": "record a premium payment",
    "policy_issue": "issue the policy",
    "claim_register": "register a claim",
    "cashless_preauth": "raise a cashless pre-authorisation",
    "endorsement_apply": "change the policy",
}


def summarise(tool_name: str, args: dict[str, Any]) -> str:
    """What the customer is being asked to agree to, in their words.

    Every declared argument is shown. A confirmation that hides a field is
    not a confirmation of the call that will actually run.
    """
    what = _LABELS.get(tool_name, tool_name.replace("_", " "))
    shown = {k: v for k, v in args.items() if not k.startswith("_")}
    if not shown:
        return f"I am about to {what}. Shall I go ahead?"
    parts = ", ".join(f"{k.replace('_', ' ')}: {v}" for k, v in shown.items())
    return f"I am about to {what} - {parts}. Shall I go ahead?"


def pending_of(state: dict) -> dict | None:
    p = state.get("pending_confirmation")
    if not p:
        return None
    if time.time() - p.get("ts", 0) > TTL_S:
        state.pop("pending_confirmation", None)
        return None
    return p


def park(state: dict, tool_name: str, args: dict[str, Any]) -> dict:
    """Record the call the customer has been asked about."""
    token = token_for(tool_name, args)
    pending = {"token": token, "tool": tool_name, "args": args,
               "summary": summarise(tool_name, args), "ts": time.time()}
    state["pending_confirmation"] = pending
    return pending


def clear(state: dict) -> None:
    state.pop("pending_confirmation", None)


def read_answer(text: str | None) -> str:
    """"yes" / "no" / "unclear" - and unclear means ask again, not proceed."""
    t = (text or "").strip()
    if AFFIRMATIVE.match(t):
        return "yes"
    if NEGATIVE.match(t):
        return "no"
    return "unclear"
