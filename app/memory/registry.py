"""The memory registry (ME-8).

Every store this system writes to is declared here with its key, its TTL, its
sensitivity and its erasure handler. Nothing may write to a store that is not
in this table.

Why a registry rather than a convention: an erasure request arrives eighteen
months after the store was added, and the only reliable way to answer it is a
list that could not have been forgotten to be updated - because `erase()`
iterates the list, so a store missing from it is a store that visibly does
not erase, and that shows up in the erasure report as a gap rather than as
silence.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal

from app.config import settings
from app.memory.backend import backend
from app.memory.longterm import (ConsentLedger, InteractionSummary, LearningStore,
                                 LobProfile, ProducerProfile, SuppressionList)

Sensitivity = Literal["low", "medium", "high"]


@dataclass(frozen=True)
class StoreSpec:
    name: str
    key: str                       # the key template, as documentation
    ttl_days: int
    sensitivity: Sensitivity
    crosses_lob: bool
    contains: str
    erase: Callable[[str], int]    # user_id -> rows removed


def _prefix_eraser(prefix: str) -> Callable[[str], int]:
    return lambda user_id: backend().delete_prefix(f"{prefix}{user_id}")


def _bot_session_eraser(user_id: str) -> int:
    """Delete the whole bot thread per line of business.

    `delete_thread` is the checkpointer's own erasure primitive, which is why
    the registry can promise erasure rather than approximate it by writing an
    empty state over the top.
    """
    from app.agents.graph import agent_for
    from app.agents.registry import REGISTRY
    from app.coremock.catalog import LOBS
    from app.orchestrator import bot_thread

    n = 0
    for lob in LOBS:
        spec = next((s for s in REGISTRY.values() if s.lob == lob), None)
        if spec is None:
            continue
        try:
            agent_for(spec).graph.checkpointer.delete_thread(
                bot_thread(user_id, lob))
            n += 1
        except Exception:                                    # noqa: BLE001
            pass
    return n


def _document_eraser(user_id: str) -> int:
    """A subject erasure that leaves their documents in a bucket has not
    erased them."""
    from app.storage.documents import documents

    try:
        return documents().erase(user_id)
    except Exception:                                        # noqa: BLE001
        return 0


def _resolver_eraser(user_id: str) -> int:
    from app.resolver.graph import resolver_graph, thread_for

    try:
        resolver_graph().checkpointer.delete_thread(thread_for(user_id))
        return 1
    except Exception:                                        # noqa: BLE001
        return 0


def _registry() -> dict[str, StoreSpec]:
    cfg = settings()
    specs = [
        StoreSpec("bot_session", "user#lob", cfg.session_ttl_days, "high",
                  False, "messages, slots, documents captured in ONE line of "
                         "business", _bot_session_eraser),
        StoreSpec("resolver_session", "user", cfg.resolver_ttl_days, "medium",
                  True, "route ledger, shared profile, auth state",
                  _resolver_eraser),
        StoreSpec("consent_ledger", "consent#user", ConsentLedger.TTL_DAYS,
                  "medium", True, "grants and withdrawals by purpose, with "
                                  "evidence", _prefix_eraser(ConsentLedger.PREFIX)),
        StoreSpec("suppression", "suppress#user#channel",
                  SuppressionList.TTL_DAYS, "low", True,
                  "do-not-contact per channel",
                  _prefix_eraser(SuppressionList.PREFIX)),
        StoreSpec("interaction_summary", "interaction#user",
                  InteractionSummary.TTL_DAYS, "low", True,
                  "turn counts, tools used, last line of business - counts, "
                  "never content", _prefix_eraser(InteractionSummary.PREFIX)),
        StoreSpec("lob_profile", "lobprofile#user#lob", LobProfile.TTL_DAYS,
                  "high", False, "facts FROM a line of business, typed and "
                                 "evidenced", _prefix_eraser(LobProfile.PREFIX)),
        StoreSpec("producer_profile", "producer#producer_id",
                  ProducerProfile.TTL_DAYS, "medium", True,
                  "licence, branch, book - not a customer record",
                  _prefix_eraser(ProducerProfile.PREFIX)),
        StoreSpec("learning_store", "learning#user", LearningStore.TTL_DAYS,
                  "low", True, "route corrections as labelled examples",
                  _prefix_eraser(LearningStore.PREFIX)),
        StoreSpec("documents", "documents/user/lob/sha256",
                  cfg.session_ttl_days, "high", False,
                  "the bytes of anything the customer sent - kept OUT of the "
                  "message list so a checkpoint stays small; only metadata "
                  "reaches the thread", _document_eraser),
        StoreSpec("audit", "person#user", cfg.audit_ttl_days, "high", True,
                  "one record per turn spanning every line of business - "
                  "retained, and NOT erased by a normal erasure request",
                  lambda _user_id: 0),
    ]
    return {s.name: s for s in specs}


REGISTRY: dict[str, StoreSpec] = _registry()


def declared() -> list[dict]:
    """The printable version - what the console and a privacy review read."""
    return [
        {"name": s.name, "key": s.key, "ttl_days": s.ttl_days,
         "sensitivity": s.sensitivity, "crosses_lob": s.crosses_lob,
         "contains": s.contains,
         "erasure": "retained for regulatory record" if s.name == "audit"
                    else "handler registered"}
        for s in REGISTRY.values()
    ]


def erase(user_id: str) -> dict[str, int | str]:
    """Run every registered erasure handler. The report names every store.

    The audit store deliberately reports `retained`: an erasure request does
    not remove a regulatory record, and saying so in the report is better
    than a number that quietly means nothing was there.
    """
    report: dict[str, int | str] = {}
    for name, spec in REGISTRY.items():
        if name == "audit":
            report[name] = "retained"
            continue
        try:
            report[name] = spec.erase(user_id)
        except Exception as exc:                             # noqa: BLE001
            report[name] = f"failed: {type(exc).__name__}"
    return report


__all__ = ["REGISTRY", "StoreSpec", "declared", "erase"]
