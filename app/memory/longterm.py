"""The long-term stores (ME-6), and the rule about what may be written (ME-7).

Six stores, each with a different lifetime and a different reason to exist:

| store              | key            | crosses LOB | why it is separate                |
|--------------------|----------------|-------------|-----------------------------------|
| consent ledger     | user           | yes         | consent is about the person       |
| suppression        | user + channel | yes         | a stop is a stop, everywhere      |
| interaction summary| user           | yes         | counts, not content               |
| LOB profile        | user # lob     | **no**      | facts FROM a line of business     |
| producer profile   | producer       | yes         | licence and book, not a customer  |
| learning store     | user           | yes         | route corrections, labelled data  |

The LOB profile is the one that must not cross, and it is keyed so that it
cannot: `user#health` and `user#motor` are different rows, the same way the
bot sessions are.

**ME-7 - every write is typed and evidenced.** A fact carries where it came
from: the customer said it, a tool returned it, the directory holds it, a
document states it. `model_inference` is not a source, and `write()` refuses
it. The reason is narrow and important: a model inferring "sounds like a
family with children" and that inference surviving into a permanent store is
how a profile acquires facts nobody ever told it.
"""
from __future__ import annotations

import time
from typing import Any, Literal

from pydantic import BaseModel, Field

from app.memory.backend import backend

EvidenceSource = Literal["user_stated", "tool_result", "directory",
                         "document", "channel", "operator"]

FORBIDDEN_SOURCE = "model_inference"


class Evidence(BaseModel):
    source: EvidenceSource
    ref: str = ""                  # tool name, chunk id, message id, doc id
    turn: int = 0
    ts: float = Field(default_factory=time.time)


class Fact(BaseModel):
    key: str
    value: Any
    evidence: Evidence


class MemoryWriteRefused(Exception):
    """ME-7: raised rather than silently dropping - a caller that tried to
    persist an inference has a bug worth failing loudly on."""


def _check(evidence: Evidence | dict) -> Evidence:
    # Checked BEFORE validation: `model_inference` is not an invalid enum
    # value to be reported as a schema error, it is a specific thing somebody
    # tried to do, and the message should say which.
    raw = (evidence.source if isinstance(evidence, Evidence)
           else evidence.get("source"))
    if raw == FORBIDDEN_SOURCE:
        raise MemoryWriteRefused(
            "a model inference about a person is not a memory write - "
            "record what was said or what a tool returned, not what was "
            "guessed from it")
    return evidence if isinstance(evidence, Evidence) else Evidence(**evidence)


# ---------------------------------------------------------------------------
class ConsentLedger:
    """Append-only. A withdrawal never deletes the grant that preceded it -
    the question "was there consent on the day we called" has to stay
    answerable."""

    PREFIX = "consent#"
    TTL_DAYS = 2555

    def record(self, user_id: str, purpose: str, granted: bool,
               evidence: Evidence | dict) -> dict:
        ev = _check(evidence)
        pk = f"{self.PREFIX}{user_id}"
        row = backend().get(pk) or {"user_id": user_id, "events": []}
        row["events"].append({"purpose": purpose, "granted": granted,
                              "evidence": ev.model_dump()})
        backend().put(pk, row, self.TTL_DAYS)
        return row

    def current(self, user_id: str) -> dict[str, bool]:
        row = backend().get(f"{self.PREFIX}{user_id}") or {"events": []}
        state: dict[str, bool] = {}
        for e in row["events"]:
            state[e["purpose"]] = e["granted"]
        return state

    def is_current(self, user_id: str, purpose: str) -> bool:
        return self.current(user_id).get(purpose, False)

    def history(self, user_id: str) -> list[dict]:
        return (backend().get(f"{self.PREFIX}{user_id}") or {}).get("events", [])


class SuppressionList:
    """A stop is a stop. Checked before any outbound, on every channel."""

    PREFIX = "suppress#"
    TTL_DAYS = 2555

    def add(self, user_id: str, channel: str, reason: str,
            evidence: Evidence | dict) -> None:
        ev = _check(evidence)
        backend().put(f"{self.PREFIX}{user_id}#{channel}",
                      {"user_id": user_id, "channel": channel,
                       "reason": reason, "evidence": ev.model_dump()},
                      self.TTL_DAYS)

    def suppressed(self, user_id: str, channel: str) -> bool:
        return backend().get(f"{self.PREFIX}{user_id}#{channel}") is not None


class InteractionSummary:
    """Counts and last-seen, not a narrative. A model-written summary of a
    person is exactly what ME-7 exists to keep out of a permanent store."""

    PREFIX = "interaction#"
    TTL_DAYS = 730

    def record_turn(self, user_id: str, lob: str | None, bot_id: str | None,
                    tools: list[str], handed_off: bool = False) -> dict:
        pk = f"{self.PREFIX}{user_id}"
        row = backend().get(pk) or {"user_id": user_id, "turns": 0,
                                    "by_lob": {}, "tools": {},
                                    "handoffs": 0, "first_seen": time.time()}
        row["turns"] += 1
        row["last_seen"] = time.time()
        row["last_lob"] = lob
        row["last_bot"] = bot_id
        if lob:
            row["by_lob"][lob] = row["by_lob"].get(lob, 0) + 1
        for t in tools:
            row["tools"][t] = row["tools"].get(t, 0) + 1
        if handed_off:
            row["handoffs"] += 1
        backend().put(pk, row, self.TTL_DAYS)
        return row

    def get(self, user_id: str) -> dict:
        return backend().get(f"{self.PREFIX}{user_id}") or {}


class LobProfile:
    """Facts FROM a line of business. Keyed `user#lob` so it cannot leak the
    way the shared profile cannot hold it."""

    PREFIX = "lobprofile#"
    TTL_DAYS = 1095

    def put(self, user_id: str, lob: str, fact: Fact | dict) -> dict:
        f = fact if isinstance(fact, Fact) else Fact(**fact)
        _check(f.evidence)
        pk = f"{self.PREFIX}{user_id}#{lob}"
        row = backend().get(pk) or {"user_id": user_id, "lob": lob, "facts": {}}
        row["facts"][f.key] = f.model_dump()
        backend().put(pk, row, self.TTL_DAYS)
        return row

    def get(self, user_id: str, lob: str) -> dict:
        row = backend().get(f"{self.PREFIX}{user_id}#{lob}") or {}
        return row.get("facts", {})


class ProducerProfile:
    PREFIX = "producer#"
    TTL_DAYS = 2555

    def put(self, producer_id: str, data: dict,
            evidence: Evidence | dict) -> None:
        ev = _check(evidence)
        backend().put(f"{self.PREFIX}{producer_id}",
                      data | {"evidence": ev.model_dump()}, self.TTL_DAYS)

    def get(self, producer_id: str) -> dict:
        return backend().get(f"{self.PREFIX}{producer_id}") or {}


class LearningStore:
    """RS-7's output. Every mis-route the next turn corrected, stored as a
    labelled example - the golden set §8.2 asks for, that nobody annotated."""

    PREFIX = "learning#"
    TTL_DAYS = 2555

    def record_correction(self, user_id: str, text: str, predicted: str | None,
                          actual: str, decided_by: str, confidence: float
                          ) -> dict:
        pk = f"{self.PREFIX}{user_id}"
        row = backend().get(pk) or {"user_id": user_id, "examples": []}
        row["examples"].append({
            "text": (text or "")[:300], "predicted": predicted,
            "actual": actual, "decided_by": decided_by,
            "confidence": confidence, "ts": time.time()})
        backend().put(pk, row, self.TTL_DAYS)
        return row

    def examples(self, user_id: str | None = None) -> list[dict]:
        if user_id:
            return (backend().get(f"{self.PREFIX}{user_id}") or {}).get(
                "examples", [])
        out: list[dict] = []
        for _, row in backend().scan_prefix(self.PREFIX):
            out.extend(row.get("examples", []))
        return out


consent = ConsentLedger()
suppression = SuppressionList()
interactions = InteractionSummary()
lob_profile = LobProfile()
producers = ProducerProfile()
learning = LearningStore()
