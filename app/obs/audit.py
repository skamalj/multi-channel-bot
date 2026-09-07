"""Audit store (OB-3): one record per turn, keyed by PERSON, spanning every
line of business.

This is the deliberate asymmetry in the design. Runtime context is
compartmented - a motor turn cannot read the health session, cannot retrieve
health documents and cannot call a health tool. The **record** is not
compartmented, because the questions asked of it are asked about a person:
what were they told, on what date, by which configuration, citing which
document version.

Separation governs runtime context. It does not govern the record. A
regulator asking "what did you tell this customer" is not asking per line of
business, and an audit store that can only answer per line of business cannot
answer them at all.

Consequences worth stating: this store is the highest-sensitivity thing here,
it is write-only from the conversation's point of view, and it is NOT erased
by a subject erasure request - `memory/registry.py` reports it as retained
rather than pretending.
"""
from __future__ import annotations

import time
from typing import Any

from app.config import settings
from app.memory.backend import backend
from app.obs.trace import Trace

PREFIX = "person#"


def _summarise(trace: Trace) -> dict[str, Any]:
    tools, gates, guardrails, retrieval = [], [], [], []
    resolution: dict[str, Any] = {}
    llm_ms = 0.0
    for ev in trace.events:
        if ev.kind == "tool":
            tools.append({"name": ev.label, "ms": ev.ms,
                          "effect": ev.detail.get("effect"),
                          "authority": ev.detail.get("authority")})
        elif ev.kind == "gate":
            gates.append({"label": ev.label, "detail": ev.detail})
        elif ev.kind == "guardrail":
            guardrails.append({"label": ev.label, "detail": ev.detail})
        elif ev.kind == "retrieve":
            retrieval.append({"label": ev.label,
                              "accepted": ev.detail.get("accepted"),
                              "rejected": ev.detail.get("rejected"),
                              "cited": ev.detail.get("cited")})
        elif ev.kind == "resolve":
            resolution[ev.label] = ev.detail
        elif ev.kind == "llm":
            llm_ms += ev.ms or 0.0
    return {"tools": tools, "gates": gates, "guardrails": guardrails,
            "retrieval": retrieval, "resolution": resolution,
            "llm_ms": round(llm_ms, 1)}


def record(user_id: str, bot_id: str | None, lob: str | None,
           persona: str | None, trace: Trace, inbound: str | None,
           outbound: str | None, citations: list[str] | None = None) -> dict:
    cfg = settings()
    entry = {
        "ts": time.time(),
        "trace_id": trace.trace_id,
        "bot_id": bot_id, "lob": lob, "persona": persona,
        "prompt_version": cfg.prompt_version,
        "config_version": cfg.config_version,
        "model": cfg.bedrock_model_id if not cfg.mock_llm else "stub",
        "inbound": (inbound or "")[:500],
        "outbound": (outbound or "")[:1000],
        "citations": citations or [],
        **_summarise(trace),
    }
    pk = f"{PREFIX}{user_id}"
    row = backend().get(pk) or {"user_id": user_id, "entries": []}
    row["entries"].append(entry)
    # Keep the tail bounded in the demo store; the real one is append-only
    # into an immutable log and never truncates.
    row["entries"] = row["entries"][-200:]
    backend().put(pk, row, cfg.audit_ttl_days)
    return entry


def for_person(user_id: str, limit: int = 50) -> list[dict]:
    """Every turn, every line of business, newest last."""
    row = backend().get(f"{PREFIX}{user_id}") or {"entries": []}
    return row["entries"][-limit:]
