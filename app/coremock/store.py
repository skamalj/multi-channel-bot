"""Seeded records behind the InsureMO-shaped interface.

Six interfaces, the shapes the real thing has: product, rating, quote,
policy, endorsement, claims. Everything a tool needs comes from here, and
nothing here knows a model exists.

Behaviours kept deliberately (CO-4), because a demo core that always answers
instantly and always succeeds teaches the wrong lesson:

* latency on every call, so a turn's timing is honest
* a **pending** state that is a real outcome, not a failure
* a **429** from the quote interface under burst - retryable
* one interface that is simply **not available** in this environment

And two controls the real core also has: idempotency on writes (CO-1) and a
gate chain that must clear before issuance (CO-5).
"""
from __future__ import annotations

import hashlib
import time
import uuid
from collections import deque
from datetime import date, datetime, timedelta, timezone

from app.config import settings
from app.coremock.catalog import gates_for, product, underwriting_rules

QUOTE_VALIDITY_DAYS = 15

# CO-4: named so nobody debugs it for an hour. Endorsement is the interface
# this environment does not expose; a tool that calls it gets a clean,
# typed "unavailable", not a stack trace.
UNAVAILABLE_INTERFACES = {"endorsement"}

# CO-4: the quote interface throttles under burst, the way a rated core does.
_RATE_WINDOW_S = 10.0
_RATE_LIMIT = 5


class CoreError(Exception):
    """A typed core failure. Tools turn these into structured tool results."""

    def __init__(self, code: str, detail: str, retryable: bool = False,
                 http_status: int = 400):
        super().__init__(detail)
        self.code, self.detail = code, detail
        self.retryable, self.http_status = retryable, http_status

    def as_result(self) -> dict:
        return {"error": self.code, "detail": self.detail,
                "retryable": self.retryable, "status": self.http_status}


# --- in-memory tables ------------------------------------------------------
_QUOTES: dict[str, dict] = {}
_IDEMPOTENCY: dict[str, dict] = {}
_APPLICATIONS: dict[str, dict] = {}
_CLAIMS: dict[str, dict] = {}
_CALLS: dict[str, deque] = {}

_CUSTOMERS = {
    "919820000009": {"customer_id": "C-10001", "name": "Priya Sharma",
                     "city": "Pune", "pincode": "411038",
                     "policies": ["PHS-4471902", "PMS-8830145"],
                     "kyc_status": "verified"},
    "919820000002": {"customer_id": "C-10002", "name": "Rohit Verma",
                     "city": "Mumbai", "pincode": "400058",
                     "policies": ["PHSR-2210033"],
                     "kyc_status": "pending"},
    "919820000003": {"customer_id": "C-10003", "name": "Anita Desai",
                     "city": "Bengaluru", "pincode": "560095",
                     "policies": [], "kyc_status": "not_started"},
}

_PRODUCERS = {
    "919820000001": {"producer_id": "P-2201", "name": "Rakesh Nair",
                     "licence": "POSP/2024/MH/88213", "branch": "Pune West",
                     "licence_valid_to": "2027-03-31"},
    "agent-demo": {"producer_id": "P-2202", "name": "Demo Producer",
                   "licence": "POSP/2024/MH/00000", "branch": "Demo",
                   "licence_valid_to": "2027-03-31"},
}

_POLICIES: dict[str, dict] = {
    "PHS-4471902": {
        "policy_id": "PHS-4471902", "lob": "health", "product_id": "PHS",
        "customer_id": "C-10001",
        "product_name": "Protec Health Secure", "sum_insured": 1000000,
        "status": "in_force", "start_date": "2025-06-14", "end_date": "2026-06-13",
        "renewed_to": "2026-06-14",
        "members": [
            {"member_id": "M1", "name": "Priya Sharma", "age": 34, "relation": "self"},
            {"member_id": "M2", "name": "Arjun Sharma", "age": 37, "relation": "spouse"},
            {"member_id": "M3", "name": "Meera Sharma", "age": 62, "relation": "mother"},
        ],
        "documents": ["policy_schedule", "e_card", "premium_receipt"],
    },
    "PMS-8830145": {
        "policy_id": "PMS-8830145", "lob": "motor", "product_id": "PMS",
        "customer_id": "C-10001",
        "product_name": "Protec Motor Shield", "idv": 480000,
        "status": "in_force", "start_date": "2026-01-08", "end_date": "2027-01-07",
        "vehicle": {"registration": "MH12AB1234", "make": "Maruti",
                    "model": "Baleno", "variant": "Zeta", "year": 2021,
                    "fuel": "Petrol", "rto": "MH12 Pune"},
        "ncb_pct": 35,
        "documents": ["policy_schedule", "premium_receipt"],
    },
    "PHSR-2210033": {
        "policy_id": "PHSR-2210033", "lob": "health", "product_id": "PHSR",
        "customer_id": "C-10002",
        "product_name": "Protec Health Senior Care", "sum_insured": 500000,
        "status": "in_force", "start_date": "2026-06-01", "end_date": "2027-05-31",
        "members": [
            {"member_id": "M1", "name": "Rohit Verma", "age": 66, "relation": "self"},
        ],
        "documents": ["policy_schedule", "e_card"],
    },
}

_VEHICLES = {
    "MH12AB1234": {"registration": "MH12AB1234", "make": "Maruti",
                   "model": "Baleno", "variant": "Zeta", "year": 2021,
                   "fuel": "Petrol", "cc": 1197, "vehicle_class": "private_car",
                   "rto": "MH12 Pune", "prior_insurer": "Protec",
                   "prior_ncb_pct": 35, "prior_expiry": "2027-01-07"},
    "KA05MN7788": {"registration": "KA05MN7788", "make": "Hyundai",
                   "model": "i20", "variant": "Sportz", "year": 2019,
                   "fuel": "Petrol", "cc": 1197, "vehicle_class": "private_car",
                   "rto": "KA05 Bengaluru", "prior_insurer": "Other",
                   "prior_ncb_pct": 20, "prior_expiry": "2026-08-02"},
    "MH14XY9021": {"registration": "MH14XY9021", "make": "Honda",
                   "model": "Activa", "variant": "6G", "year": 2020,
                   "fuel": "Petrol", "cc": 109, "vehicle_class": "two_wheeler",
                   "rto": "MH14 Pimpri", "prior_insurer": "Other",
                   "prior_ncb_pct": 25, "prior_expiry": "2026-05-19"},
    "DL8CAF4412": {"registration": "DL8CAF4412", "make": "Toyota",
                   "model": "Innova Crysta", "variant": "GX", "year": 2014,
                   "fuel": "Diesel", "cc": 2393, "vehicle_class": "private_car",
                   "rto": "DL8C Delhi", "prior_insurer": "Other",
                   "prior_ncb_pct": 0, "prior_expiry": "2026-04-30"},
}

_NETWORK = {
    "hospital": [
        {"name": "Sahyadri Super Speciality", "pincode": "411004", "city": "Pune",
         "specialties": ["orthopaedics", "cardiology"], "cashless": True, "km": 2.4},
        {"name": "Ruby Hall Clinic", "pincode": "411001", "city": "Pune",
         "specialties": ["orthopaedics", "oncology"], "cashless": True, "km": 4.1},
        {"name": "Jehangir Hospital", "pincode": "411001", "city": "Pune",
         "specialties": ["general", "paediatrics"], "cashless": True, "km": 4.8},
        {"name": "Kokilaben Dhirubhai Ambani", "pincode": "400053", "city": "Mumbai",
         "specialties": ["cardiology", "oncology", "neurology"], "cashless": True,
         "km": 3.2},
        {"name": "Manipal Hospital Sarjapur", "pincode": "560035", "city": "Bengaluru",
         "specialties": ["general", "orthopaedics"], "cashless": True, "km": 5.6},
    ],
    "garage": [
        {"name": "AutoCare Kothrud", "pincode": "411038", "city": "Pune",
         "cashless": True, "km": 1.9},
        {"name": "Speedworks Baner", "pincode": "411045", "city": "Pune",
         "cashless": True, "km": 6.3},
        {"name": "Sai Motors Andheri", "pincode": "400053", "city": "Mumbai",
         "cashless": True, "km": 2.1},
    ],
}


# --- CO-4 behaviours -------------------------------------------------------
def _latency(interface: str) -> None:
    cfg = settings()
    if not cfg.core_chaos:
        return
    if interface in UNAVAILABLE_INTERFACES:
        raise CoreError("interface_unavailable",
                        f"the {interface} interface is not exposed in this "
                        f"environment; raise a service request to enable it",
                        retryable=False, http_status=503)
    time.sleep(cfg.core_latency_ms / 1000)


def _rate_limit(interface: str) -> None:
    """A rated core throttles. Retryable, and the tool result says so."""
    if not settings().core_chaos:
        return
    now = time.monotonic()
    window = _CALLS.setdefault(interface, deque())
    while window and now - window[0] > _RATE_WINDOW_S:
        window.popleft()
    if len(window) >= _RATE_LIMIT:
        raise CoreError("rate_limited",
                        f"{interface} interface: more than {_RATE_LIMIT} calls "
                        f"in {int(_RATE_WINDOW_S)}s - retry shortly",
                        retryable=True, http_status=429)
    window.append(now)


def reset_chaos() -> None:
    """Tests drive the throttle deliberately; they also have to clear it."""
    _CALLS.clear()


# --- idempotency (CO-1) ----------------------------------------------------
def _idem(key: str | None, op: str, payload: dict):
    """Returns a cached result for a replayed write, or None.

    The key is scoped by operation and by payload digest: the same key with a
    different body is a client bug and must not silently return the old
    answer.
    """
    if not key:
        return None
    digest = hashlib.sha256(
        repr(sorted(payload.items())).encode("utf-8")).hexdigest()[:16]
    hit = _IDEMPOTENCY.get(f"{op}:{key}")
    if hit is None:
        return None
    if hit["digest"] != digest:
        raise CoreError("idempotency_key_reused",
                        "this idempotency key was used with a different payload",
                        http_status=409)
    return hit["result"] | {"idempotent_replay": True}


def _remember(key: str | None, op: str, payload: dict, result: dict) -> dict:
    if key:
        digest = hashlib.sha256(
            repr(sorted(payload.items())).encode("utf-8")).hexdigest()[:16]
        _IDEMPOTENCY[f"{op}:{key}"] = {"digest": digest, "result": result}
    return result


# --- directory -------------------------------------------------------------
def customer_for(user_id: str) -> dict | None:
    return _CUSTOMERS.get(user_id)


def producer_for(user_id: str) -> dict | None:
    return _PRODUCERS.get(user_id)


def holdings_for(user_id: str) -> dict[str, bool]:
    """Facts ABOUT a line of business - which is exactly what may be shared."""
    cust = _CUSTOMERS.get(user_id)
    if not cust:
        return {}
    held: dict[str, bool] = {}
    for pid in cust["policies"]:
        pol = _POLICIES.get(pid)
        if pol:
            held[pol["lob"]] = True
    return held


def policies_for_customer(customer_id: str, lob: str) -> list[dict]:
    return [p for p in _POLICIES.values()
            if p["customer_id"] == customer_id and p["lob"] == lob]


# --- quote interface -------------------------------------------------------
def save_quote(q: dict, idempotency_key: str | None = None) -> dict:
    _rate_limit("quote")
    _latency("quote")
    payload = {k: v for k, v in q.items()
               if k in ("product_id", "sum_insured", "idv", "net_premium")}
    cached = _idem(idempotency_key, "quote_create", payload)
    if cached:
        return cached

    qid = f"Q-{uuid.uuid4().hex[:8].upper()}"
    now = datetime.now(timezone.utc)
    q |= {"quote_id": qid,
          "created_at": now.isoformat(),
          "valid_until": (now + timedelta(days=QUOTE_VALIDITY_DAYS)).date().isoformat(),
          "status": "rated"}
    _QUOTES[qid] = q
    return _remember(idempotency_key, "quote_create", payload, q)


def get_quote(quote_id: str) -> dict:
    _latency("quote")
    q = _QUOTES.get(quote_id)
    if not q:
        return {"error": "not_found", "quote_id": quote_id}
    expired = date.fromisoformat(q["valid_until"]) < datetime.now(timezone.utc).date()
    # An expired quote is re-priced by the core, never resurrected.
    return q | {"expired": expired,
                "next_step": "re-rate" if expired else "proceed_to_application"}


# --- policy interface ------------------------------------------------------
def get_policy(policy_id: str, lob: str) -> dict:
    _latency("policy")
    p = _POLICIES.get(policy_id)
    if not p:
        return {"error": "not_found", "policy_id": policy_id}
    if p["lob"] != lob:
        # Tools are the third leak path: a policy read must be scoped to the
        # active line of business even when the session already is.
        return {"error": "out_of_scope",
                "detail": f"policy {policy_id} is not a {lob} policy"}
    return p


def waiting_periods(policy_id: str) -> dict:
    """Waiting-period status per policy, answered from the wording that was in
    force on the POLICY start date - not the one in force today (KB-6)."""
    _latency("policy")
    p = _POLICIES.get(policy_id)
    if not p or p["lob"] != "health":
        return {"error": "not_found", "policy_id": policy_id}

    as_of = p["start_date"]
    prod = product(p["product_id"], as_of=as_of)
    wp = prod["waiting_periods"]
    start = date.fromisoformat(as_of)
    today = datetime.now(timezone.utc).date()
    months_elapsed = (today.year - start.year) * 12 + today.month - start.month

    items = []
    for kind, value in wp.items():
        months = value if kind.endswith("months") else round(value / 30, 2)
        eligible = start + timedelta(days=round(30.44 * months))
        items.append({
            "kind": kind,
            "required_months": months,
            "served": today >= eligible,
            "eligible_from": eligible.isoformat(),
        })
    return {
        "policy_id": policy_id,
        "product_id": p["product_id"],
        "start_date": as_of,
        "months_elapsed": months_elapsed,
        "wording_version": prod.get("wording_version"),
        "wording_note": prod.get("wording_note"),
        "items": items,
    }


# --- application / gate chain (CO-5) --------------------------------------
def _new_application(quote_id: str, customer_id: str, lob: str,
                     user_id: str = "") -> dict:
    app_id = f"A-{uuid.uuid4().hex[:8].upper()}"
    app = {
        "application_id": app_id, "quote_id": quote_id,
        "customer_id": customer_id,
        # Who opened it. A customer id is only known for a seeded customer,
        # so the identity that actually made the call is what an entitlement
        # check can rely on.
        "user_id": user_id, "lob": lob, "status": "open",
        "gates": [g | {"state": "not_started"} for g in gates_for(lob)],
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    _APPLICATIONS[app_id] = app
    return app


def application_start(quote_id: str, customer_id: str,
                      user_id: str = "",
                      idempotency_key: str | None = None) -> dict:
    _latency("policy")
    q = _QUOTES.get(quote_id)
    if not q:
        return {"error": "not_found", "quote_id": quote_id}
    if date.fromisoformat(q["valid_until"]) < datetime.now(timezone.utc).date():
        return {"error": "quote_expired", "quote_id": quote_id,
                "detail": "re-rate the quote; an expired quote is never resurrected"}
    cached = _idem(idempotency_key, "application_start", {"quote_id": quote_id})
    if cached:
        return cached
    app = _new_application(quote_id, customer_id, q["lob"], user_id)
    return _remember(idempotency_key, "application_start",
                     {"quote_id": quote_id}, app)


def application_get(application_id: str) -> dict:
    _latency("policy")
    return _APPLICATIONS.get(application_id) or {
        "error": "not_found", "application_id": application_id}


def _set_gate(app: dict, gate_id: str, state: str, detail: str = "") -> None:
    for g in app["gates"]:
        if g["id"] == gate_id:
            g["state"] = state
            if detail:
                g["detail"] = detail
            return
    raise CoreError("unknown_gate", f"{gate_id} is not a gate on this application")


def kyc_verify(application_id: str, document_type: str,
               reference: str) -> dict:
    """CO-5 gate 1. Deterministic on the reference so a demo is repeatable."""
    _latency("kyc")
    app = _APPLICATIONS.get(application_id)
    if not app:
        return {"error": "not_found", "application_id": application_id}
    if document_type not in ("pan", "aadhaar_offline_xml", "ckyc", "passport"):
        return {"error": "unsupported_document", "document_type": document_type}
    # A reference ending in 0 lands in manual review - a PENDING state, which
    # is an outcome, not a failure.
    if reference.strip().endswith("0"):
        _set_gate(app, "kyc", "pending", "referred to manual verification")
        return {"application_id": application_id, "gate": "kyc",
                "state": "pending", "sla_hours": 24,
                "detail": "referred to manual verification"}
    _set_gate(app, "kyc", "cleared", f"{document_type} verified")
    return {"application_id": application_id, "gate": "kyc", "state": "cleared"}


def underwrite(application_id: str, declared_conditions: list[str] | None = None,
               oldest_member_age: int | None = None) -> dict:
    """CO-5 gate 2 for health. The rules live in products.yaml; the model
    never decides eligibility, it reads this decision back."""
    _latency("uw")
    app = _APPLICATIONS.get(application_id)
    if not app:
        return {"error": "not_found", "application_id": application_id}
    rules = underwriting_rules(app["lob"])
    conditions = [c.lower().strip() for c in (declared_conditions or [])]
    q = _QUOTES.get(app["quote_id"], {})

    for c in conditions:
        if c in rules.get("auto_decline_conditions", []):
            _set_gate(app, "uw", "declined", f"declined on {c}")
            return {"application_id": application_id, "gate": "uw",
                    "state": "declined", "reason": c,
                    "referral_note": "a declinature is communicated by the "
                                     "underwriter, not by this assistant"}

    reasons = []
    age = oldest_member_age or max(
        (m["member_age"] for m in q.get("member_lines", [])), default=0)
    if age >= rules.get("ppmc_age_threshold", 999) and any(
            c in rules.get("ppmc_declared_conditions", []) for c in conditions):
        reasons.append("pre-policy medical check-up required")
    if q.get("sum_insured", 0) > rules.get("referral_sum_insured", 10**9) and conditions:
        reasons.append("sum insured above the straight-through limit with a "
                       "declared condition")
    if reasons:
        _set_gate(app, "uw", "referred", "; ".join(reasons))
        return {"application_id": application_id, "gate": "uw",
                "state": "referred", "reasons": reasons, "sla_hours": 48}
    _set_gate(app, "uw", "cleared", "straight-through")
    return {"application_id": application_id, "gate": "uw", "state": "cleared"}


def inspection_schedule(application_id: str, slot: str) -> dict:
    """CO-5 gate 2 for motor."""
    _latency("inspection")
    app = _APPLICATIONS.get(application_id)
    if not app:
        return {"error": "not_found", "application_id": application_id}
    _set_gate(app, "inspection", "pending", f"surveyor slot {slot}")
    return {"application_id": application_id, "gate": "inspection",
            "state": "pending", "slot": slot, "sla_hours": 24}


def inspection_result(application_id: str, outcome: str,
                      note: str = "") -> dict:
    """The surveyor's report coming back.

    Without this the inspection gate can only ever be `pending`, so a motor
    policy could never issue - the chain was missing its second half rather
    than being strict. A customer cannot clear their own inspection, which is
    why the tool that calls this is producer-only.
    """
    _latency("inspection")
    app = _APPLICATIONS.get(application_id)
    if not app:
        return {"error": "not_found", "application_id": application_id}
    if outcome not in ("clean", "damage_noted", "declined"):
        return {"error": "bad_outcome", "outcome": outcome,
                "expected": ["clean", "damage_noted", "declined"]}
    if outcome == "clean":
        _set_gate(app, "inspection", "cleared", note or "surveyor report clean")
        return {"application_id": application_id, "gate": "inspection",
                "state": "cleared", "outcome": outcome}
    _set_gate(app, "inspection", "declined" if outcome == "declined"
              else "pending", note or outcome)
    return {"application_id": application_id, "gate": "inspection",
            "state": "declined" if outcome == "declined" else "pending",
            "outcome": outcome, "note": note}


def payment_collect(application_id: str, amount: float, mode: str,
                    idempotency_key: str | None = None) -> dict:
    """CO-5 gate 3. Money is the one place a duplicate is unforgivable, so
    this is the strictest use of the idempotency key in the mock."""
    _latency("payment")
    app = _APPLICATIONS.get(application_id)
    if not app:
        return {"error": "not_found", "application_id": application_id}
    payload = {"application_id": application_id, "amount": round(amount, 2)}
    cached = _idem(idempotency_key, "payment_collect", payload)
    if cached:
        return cached
    if mode not in ("upi", "netbanking", "card", "cheque"):
        return {"error": "unsupported_mode", "mode": mode}
    if mode == "cheque":
        _set_gate(app, "payment", "pending", "cheque realisation")
        result = {"application_id": application_id, "gate": "payment",
                  "state": "pending", "detail": "awaiting cheque realisation",
                  "sla_hours": 72}
    else:
        _set_gate(app, "payment", "cleared", f"{mode} realised")
        result = {"application_id": application_id, "gate": "payment",
                  "state": "cleared",
                  "receipt_id": f"R-{uuid.uuid4().hex[:8].upper()}",
                  "amount": round(amount, 2)}
    return _remember(idempotency_key, "payment_collect", payload, result)


def policy_issue(application_id: str,
                 idempotency_key: str | None = None) -> dict:
    """CO-5: issuance is refused until every blocking gate has cleared.

    The refusal names the gate. A bot that says "your policy is issued" while
    a gate is pending is the single most expensive sentence in this domain.
    """
    _latency("policy")
    app = _APPLICATIONS.get(application_id)
    if not app:
        return {"error": "not_found", "application_id": application_id}

    blocking = [g for g in app["gates"]
                if g.get("blocking") and g["state"] != "cleared"]
    if blocking:
        return {"error": "gate_not_cleared",
                "application_id": application_id,
                "blocked_by": [{"id": g["id"], "name": g["name"],
                                "state": g["state"],
                                "detail": g.get("detail", "")} for g in blocking],
                "detail": "issuance refused until every blocking gate clears"}

    cached = _idem(idempotency_key, "policy_issue",
                   {"application_id": application_id})
    if cached:
        return cached

    q = _QUOTES.get(app["quote_id"], {})
    pid = f"{q.get('product_id', 'PXX')}-{uuid.uuid4().hex[:7].upper()}"
    today = datetime.now(timezone.utc).date()
    policy = {
        "policy_id": pid, "lob": app["lob"], "product_id": q.get("product_id"),
        "product_name": q.get("product_name"),
        "customer_id": app["customer_id"], "status": "in_force",
        "start_date": today.isoformat(),
        "end_date": (today + timedelta(days=364)).isoformat(),
        "gross_premium": q.get("gross_premium"),
        "documents": ["policy_schedule", "premium_receipt"],
    }
    if app["lob"] == "health":
        policy["sum_insured"] = q.get("sum_insured")
    else:
        policy["idv"] = q.get("idv")
    _POLICIES[pid] = policy
    app["status"] = "issued"
    app["policy_id"] = pid
    return _remember(idempotency_key, "policy_issue",
                     {"application_id": application_id}, policy)


# --- endorsement interface (CO-4: the one that is not available) -----------
def endorsement_apply(policy_id: str, endorsement_type: str,
                      value: str) -> dict:
    _latency("endorsement")            # raises interface_unavailable
    return {"policy_id": policy_id, "endorsement_type": endorsement_type,
            "value": value, "status": "applied"}       # pragma: no cover


# --- claims interface ------------------------------------------------------
def claim_register(policy_id: str, lob: str, claim_type: str,
                   incident_date: str, description: str,
                   idempotency_key: str | None = None) -> dict:
    _latency("claims")
    p = _POLICIES.get(policy_id)
    if not p:
        return {"error": "not_found", "policy_id": policy_id}
    if p["lob"] != lob:
        return {"error": "out_of_scope",
                "detail": f"policy {policy_id} is not a {lob} policy"}
    payload = {"policy_id": policy_id, "incident_date": incident_date,
               "claim_type": claim_type}
    cached = _idem(idempotency_key, "claim_register", payload)
    if cached:
        return cached

    cid = f"CLM-{uuid.uuid4().hex[:8].upper()}"
    prod = product(p["product_id"], as_of=p["start_date"])
    claim = {
        "claim_id": cid, "policy_id": policy_id, "lob": lob,
        "claim_type": claim_type, "incident_date": incident_date,
        "description": description[:500],
        # Registration is always PENDING - a claim is never "approved" by the
        # interface that opens it, and never by a model.
        "status": "registered",
        "sub_status": "pending_documents",
        "documents_required": prod.get("claim_documents", []),
        "documents_received": [],
        "tat_days": prod.get("od_claim_tat_days", 15),
        "registered_at": datetime.now(timezone.utc).isoformat(),
    }
    _CLAIMS[cid] = claim
    return _remember(idempotency_key, "claim_register", payload, claim)


def claim_status(claim_id: str) -> dict:
    _latency("claims")
    return _CLAIMS.get(claim_id) or {"error": "not_found", "claim_id": claim_id}


def cashless_preauth(policy_id: str, hospital: str, estimate: float,
                     idempotency_key: str | None = None) -> dict:
    """A pre-auth is a PENDING outcome by design: the hospital and the TPA
    both have to answer before anyone knows the number."""
    _latency("claims")
    p = _POLICIES.get(policy_id)
    if not p or p["lob"] != "health":
        return {"error": "not_found", "policy_id": policy_id}
    payload = {"policy_id": policy_id, "hospital": hospital,
               "estimate": round(estimate, 2)}
    cached = _idem(idempotency_key, "cashless_preauth", payload)
    if cached:
        return cached
    result = {
        "preauth_id": f"PA-{uuid.uuid4().hex[:8].upper()}",
        "policy_id": policy_id, "hospital": hospital,
        "estimate": round(estimate, 2),
        "status": "pending",
        "sub_status": "awaiting_hospital_confirmation",
        "sla_hours": 4,
        "note": ("an approved amount is issued by the TPA after the hospital "
                 "confirms; no amount is committed at this stage"),
    }
    return _remember(idempotency_key, "cashless_preauth", payload, result)


# --- reference data --------------------------------------------------------
def network_search(kind: str, pincode: str, specialty: str | None,
                   limit: int) -> list[dict]:
    _latency("network")
    rows = _NETWORK.get(kind, [])
    if specialty:
        rows = [r for r in rows if specialty.lower() in
                [s.lower() for s in r.get("specialties", [])]]
    # Same city first, then distance - a Pune PIN should not lead with Mumbai.
    prefix = (pincode or "")[:3]
    rows = sorted(rows, key=lambda r: (not r["pincode"].startswith(prefix),
                                       r["km"]))
    return rows[:limit]


def vehicle_lookup(registration: str) -> dict:
    _latency("vehicle")
    key = registration.replace(" ", "").replace("-", "").upper()
    v = _VEHICLES.get(key)
    if not v:
        return {"error": "not_found", "registration": registration,
                "detail": "not in the RTO extract; capture the details manually"}
    age = datetime.now(timezone.utc).date().year - v["year"]
    return v | {"vehicle_age_years": age}


def commission(producer_id: str, period: str) -> dict:
    _latency("commission")
    if not period or len(period) != 7 or period[4] != "-":
        return {"error": "bad_period", "detail": "period must be YYYY-MM"}
    seed = int(hashlib.sha256(f"{producer_id}{period}".encode()).hexdigest()[:6], 16)
    policies = 25 + seed % 30
    gwp = 900_000 + (seed % 700) * 1000
    return {"producer_id": producer_id, "period": period,
            "policies": policies, "gwp": gwp,
            "commission": round(gwp * 0.15, 2),
            "status": "payable", "payout_date": "2026-09-20"}
