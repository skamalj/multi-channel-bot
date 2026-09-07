"""Memory: what may be shared, what must be evidenced, and what erasure
actually does."""
import pytest

from app.memory import profile as profile_mem
from app.memory.longterm import (Evidence, MemoryWriteRefused, consent,
                                 interactions, lob_profile, suppression)
from app.memory.registry import REGISTRY, declared, erase
from app.orchestrator import bot_thread
from app.resolver.store import ResolverSession

EV = {"source": "user_stated", "ref": "msg-1", "turn": 1}


def test_the_shared_profile_holds_facts_about_a_line_not_from_one():
    """ME-3. holdings={"health": True} yes. has_diabetes never."""
    s = ResolverSession(user_id="u1")
    profile_mem.update(s, city="Pune", holdings={"health": True})
    assert s.profile.city == "Pune"
    with pytest.raises(ValueError, match="not shareable"):
        profile_mem.update(s, has_diabetes=True)


def test_sessions_are_keyed_apart_by_line_of_business():
    """ME-2. Isolation is a fact about addressing - the LangGraph thread id
    is the compartment boundary, so the leak requires deliberately
    constructing another thread's id."""
    assert bot_thread("u", "health") != bot_thread("u", "motor")
    assert bot_thread("u", "health") == "u#health"


def test_the_two_threads_are_keyed_differently():
    """ME-1 vs ME-2: the resolver thread is a PERSON, the bot thread is a
    person and a line of business."""
    from app.resolver.graph import thread_for

    assert thread_for("u") == "u"
    assert bot_thread("u", "health").startswith("u#")


def test_a_model_inference_about_a_person_is_not_a_memory_write():
    """ME-7. The write is refused loudly rather than dropped quietly."""
    with pytest.raises(MemoryWriteRefused):
        consent.record("u1", "marketing", True, {"source": "model_inference"})


def test_every_long_term_write_carries_evidence():
    lob_profile.put("u1", "health", {
        "key": "preferred_hospital", "value": "Ruby Hall",
        "evidence": Evidence(source="user_stated", ref="msg-9", turn=3)})
    stored = lob_profile.get("u1", "health")["preferred_hospital"]
    assert stored["evidence"]["source"] == "user_stated"
    assert stored["evidence"]["ref"] == "msg-9"


def test_a_lob_profile_does_not_cross_the_compartment():
    lob_profile.put("u1", "health", {"key": "ped_declared", "value": True,
                                     "evidence": EV})
    assert lob_profile.get("u1", "motor") == {}


def test_the_consent_ledger_is_append_only():
    """"Was there consent on the day we called" has to stay answerable."""
    consent.record("u1", "marketing", True, EV)
    consent.record("u1", "marketing", False, EV)
    assert consent.is_current("u1", "marketing") is False
    assert len(consent.history("u1")) == 2
    assert consent.history("u1")[0]["granted"] is True


def test_a_suppression_is_checked_per_channel():
    suppression.add("u1", "whatsapp", "customer asked to stop", EV)
    assert suppression.suppressed("u1", "whatsapp") is True
    assert suppression.suppressed("u1", "webchat") is False


def test_the_interaction_summary_holds_counts_not_a_narrative():
    interactions.record_turn("u1", "health", "BOT-05", ["kb_search_health"])
    interactions.record_turn("u1", "motor", "BOT-06", ["vehicle_lookup"])
    row = interactions.get("u1")
    assert row["turns"] == 2 and row["by_lob"] == {"health": 1, "motor": 1}
    assert row["last_lob"] == "motor"
    # No free text about the person anywhere in the record.
    assert set(row) <= {"user_id", "turns", "by_lob", "tools", "handoffs",
                        "first_seen", "last_seen", "last_lob", "last_bot"}


def test_every_store_is_declared_with_a_key_ttl_and_sensitivity():
    """ME-8. A store missing from the registry is a store that visibly does
    not erase."""
    for row in declared():
        assert row["key"] and row["ttl_days"] > 0
        assert row["sensitivity"] in ("low", "medium", "high")
        assert row["contains"] and row["erasure"]
    assert {"bot_session", "resolver_session", "consent_ledger",
            "suppression", "interaction_summary", "lob_profile",
            "producer_profile", "learning_store", "documents",
            "audit"} == set(REGISTRY)


def test_erasure_reports_every_store_and_says_what_it_retained():
    consent.record("u1", "marketing", True, EV)
    lob_profile.put("u1", "health", {"key": "k", "value": 1, "evidence": EV})
    interactions.record_turn("u1", "health", "BOT-05", [])

    report = erase("u1")
    assert set(report) == set(REGISTRY)
    assert report["audit"] == "retained"
    assert consent.current("u1") == {}
    assert lob_profile.get("u1", "health") == {}
    assert interactions.get("u1") == {}


def test_the_ttls_differ_because_the_stores_do():
    """ME-5. The ledger improves with age; the bot session holds the most
    sensitive data and should not."""
    assert REGISTRY["bot_session"].ttl_days < REGISTRY["resolver_session"].ttl_days
    assert REGISTRY["audit"].ttl_days > REGISTRY["resolver_session"].ttl_days
    assert REGISTRY["bot_session"].crosses_lob is False
    assert REGISTRY["lob_profile"].crosses_lob is False
