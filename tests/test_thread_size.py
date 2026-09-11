"""Keeping a thread inside what a checkpoint can hold.

DynamoDB returns at most 1 MB per query. Pagination is a safety net; the
design intent is that a text conversation never approaches the cap. Two
rules do that, and these are the tests for them:

1. Documents never enter the message list - object storage holds the bytes,
   the thread holds metadata.
2. Old turns are reduced to a rolling summary.
"""
from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from app.agents.graph import agent_for
from app.agents.registry import BOT_05
from app.obs.trace import Trace
from app.orchestrator import bot_thread

CUSTOMER = "919820000009"


def _values(user_id, lob="health"):
    return dict(agent_for(BOT_05).graph.get_state(
        {"configurable": {"thread_id": bot_thread(user_id, lob)}}).values or {})


def _say(client, user, text, **kw):
    r = client.post("/api/chat", json={"user_id": user, "text": text, **kw})
    assert r.status_code == 200, r.text
    return r.json()


def _turns(n):
    """A conversation with tool calls, which is what makes pruning hard."""
    msgs = []
    for i in range(n):
        msgs.append(HumanMessage(content=f"question {i}", id=f"h{i}"))
        msgs.append(AIMessage(content="", id=f"a{i}", tool_calls=[
            {"name": "kb_search_health", "args": {"query": "x"}, "id": f"tc{i}"}]))
        msgs.append(ToolMessage(content=f"chunk text {i}" * 20, id=f"t{i}",
                                tool_call_id=f"tc{i}"))
        msgs.append(AIMessage(content=f"answer {i}", id=f"r{i}"))
    return msgs


# --- reduction -------------------------------------------------------------
def test_a_long_thread_is_pruned_to_the_recent_window():
    from app.agents.reduce import reduce_thread
    from app.config import settings

    cfg = settings()
    messages = _turns(20)
    update = reduce_thread({"messages": messages}, Trace())
    assert update, "a 80-message thread was not reduced"

    removed = {m.id for m in update["messages"]
               if type(m).__name__ == "RemoveMessage"}
    survivors = [m for m in messages if m.id not in removed]
    assert len(survivors) <= cfg.reduce_after_messages


def test_pruning_never_orphans_a_tool_result():
    """The constraint Bedrock enforces: a toolResult with no toolUse is
    rejected outright. `cascade_tool_messages` is why this holds - a naive
    "keep the last N" breaks it about half the time."""
    from app.agents.reduce import reduce_thread

    messages = _turns(20)
    update = reduce_thread({"messages": messages}, Trace())
    removed = {m.id for m in update["messages"]
               if type(m).__name__ == "RemoveMessage"}
    survivors = [m for m in messages if m.id not in removed]

    call_ids = {tc["id"] for m in survivors
                if isinstance(m, AIMessage) for tc in (m.tool_calls or [])}
    orphans = [m.tool_call_id for m in survivors
               if isinstance(m, ToolMessage) and m.tool_call_id not in call_ids]
    assert orphans == [], f"orphaned tool results: {orphans}"


def test_a_short_thread_is_left_alone():
    from app.agents.reduce import reduce_thread

    assert reduce_thread({"messages": _turns(2)}, Trace()) == {}


def test_a_long_conversation_keeps_the_checkpoint_small(client):
    """The end-to-end property: thirty turns through the real API, and the
    persisted state stays far inside what a DynamoDB query returns."""
    import json

    user = "thread-size-01"
    for i in range(30):
        _say(client, user, f"what is the waiting period question {i}",
             lob="health")

    values = _values(user)
    size = len(json.dumps(
        [str(getattr(m, "content", "")) for m in values["messages"]]))
    assert len(values["messages"]) <= 40, len(values["messages"])
    assert size < 200_000, f"thread grew to {size} bytes"


# --- documents -------------------------------------------------------------
def test_a_document_goes_to_storage_and_only_metadata_comes_back():
    from app.storage.documents import documents

    content = b"%PDF-1.7 " + b"x" * 500_000        # half a megabyte
    ref = documents().put("u1", "health", "schedule.pdf",
                          "application/pdf", content)

    assert ref.size == len(content)
    assert ref.doc_id.startswith("DOC-")
    assert "schedule.pdf" in ref.as_message()
    # The line that reaches the thread is metadata, not the document.
    assert len(ref.as_message()) < 200
    assert "%PDF" not in ref.as_message()
    # And the bytes are retrievable by reference.
    assert documents().get(ref) == content


def test_the_same_document_twice_is_stored_once():
    """Addressed by content, so a resend is not a second copy."""
    from app.storage.documents import documents

    a = documents().put("u1", "health", "a.pdf", "application/pdf", b"same")
    b = documents().put("u1", "health", "b.pdf", "application/pdf", b"same")
    assert a.key == b.key and a.sha256 == b.sha256


def test_an_oversized_document_is_refused():
    from app.storage.documents import MAX_BYTES, documents

    import pytest

    with pytest.raises(ValueError, match="limit"):
        documents().put("u1", "health", "big.bin", "application/octet-stream",
                        b"x" * (MAX_BYTES + 1))


def test_erasure_reaches_the_documents(client):
    """ME-8: a subject erasure that leaves their documents in a bucket has
    not erased them."""
    from app.storage.documents import documents

    ref = documents().put(CUSTOMER, "health", "s.pdf", "application/pdf",
                          b"content")
    assert documents().get(ref) == b"content"
    assert documents().erase(CUSTOMER) >= 1
    assert documents().get(ref) is None


# The two tests that used to live here drove `Agent._model` directly, to prove
# a second call-time window did not re-trim through a tool pair after the
# reducer had kept it whole. That window is gone: there is no `_model` node,
# the reducer (before_model) is the ONLY thing that trims, and
# `test_pruning_never_orphans_a_tool_result` above is the invariant. Nothing
# assembles a second message list to check.


