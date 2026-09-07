"""Trace events. Everything the glass box renders comes from here.

One Trace per turn. The web UI reads it back with the reply, which is why
the console can show retrieval scores and tool latencies without a second
round trip.
"""
from __future__ import annotations

import time
import uuid
from typing import Any, Literal

from pydantic import BaseModel, Field

EventKind = Literal[
    "channel", "resolve", "bind", "route", "retrieve",
    "tool", "gate", "guardrail", "llm", "respond", "error",
]


class TraceEvent(BaseModel):
    seq: int
    kind: EventKind
    label: str
    detail: dict[str, Any] = Field(default_factory=dict)
    ms: float | None = None
    ts: float = Field(default_factory=time.time)


class Trace(BaseModel):
    trace_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    events: list[TraceEvent] = Field(default_factory=list)
    started: float = Field(default_factory=time.time)

    # kind and label are positional-ONLY: detail keys come from the domain
    # ("kind" of an inbound message, a "label" on a gate), and a collision
    # between a trace field and a payload field should not be a TypeError
    # discovered in production.
    def add(self, kind: EventKind, label: str, /, **detail: Any) -> TraceEvent:
        ev = TraceEvent(seq=len(self.events), kind=kind, label=label, detail=detail)
        self.events.append(ev)
        return ev

    def timed(self, kind: EventKind, label: str, /, **detail: Any) -> "_Timer":
        return _Timer(self, kind, label, detail)


class _Timer:
    def __init__(self, trace: Trace, kind: EventKind, label: str, detail: dict):
        self.trace, self.kind, self.label, self.detail = trace, kind, label, detail

    def __enter__(self):
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        ev = self.trace.add(self.kind, self.label, **self.detail)
        ev.ms = round((time.perf_counter() - self.t0) * 1000, 1)
        return False

    def note(self, **detail: Any) -> None:
        self.detail.update(detail)
