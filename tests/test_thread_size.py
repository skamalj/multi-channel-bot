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


def test_the_injected_summary_is_not_replayed_as_the_models_own_words():
    """A summary is context, never evidence - and never something the model
    said. Marked system_authored, it stays out of `model_visible`."""
    from app.agents.graph import model_visible
    from app.agents.reduce import _summary_messages

    pair = _summary_messages("customer wants a health quote for two adults", 40)
    assert all(m.additional_kwargs.get("system_authored") for m in pair)
    assert model_visible(pair) == []


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


def test_what_reaches_the_model_keeps_every_tool_call_with_its_result():
    """The reducer got this right and a second window downstream undid it.

    `_model` used to slice `history[-msg_history_to_keep:]` on top of the
    reduction. That cut at 12 while the reducer prunes at 24, so between the
    two numbers the slice was the only thing trimming - and it trimmed
    blindly, straight through an AI tool_calls message and the ToolMessage
    answering it. Bedrock rejects that outright:

        Expected toolResult blocks at messages.0.content for the following
        Ids: functions.kb_search_health:0

    The reducer owns the window now. This test watches the messages actually
    handed to the model, because that is where the damage was done.
    """
    agent = agent_for(BOT_05)
    seen: dict = {}

    class _Capture:
        def invoke(self, messages):
            seen["messages"] = messages
            return AIMessage(content="ok", id="final")

    original, agent.llm = agent.llm, _Capture()
    try:
        agent._model({"messages": _turns(5), "rounds": 0},
                     {"configurable": {"trace": Trace()}})
    finally:
        agent.llm = original

    sent = seen["messages"]
    call_ids = {tc["id"] for m in sent
                if isinstance(m, AIMessage) for tc in (m.tool_calls or [])}
    orphans = [m.tool_call_id for m in sent
               if isinstance(m, ToolMessage) and m.tool_call_id not in call_ids]
    assert orphans == [], f"tool results with no call: {orphans}"

    # And the whole conversation arrived - nothing was quietly dropped on the
    # way to the model.
    assert len([m for m in sent if isinstance(m, ToolMessage)]) == 5


def test_a_thread_that_does_not_divide_evenly_still_reaches_the_model_whole():
    """The shape that actually broke in production.

    A fixed-size window only orphans a tool result when the cut lands
    mid-turn, which depends on how many messages happen to precede it - so
    the bug hid behind conversations that divided evenly and appeared on the
    ones that did not. Twenty-two messages puts the cut squarely on a
    ToolMessage whose AI tool_calls message falls outside it.
    """
    agent = agent_for(BOT_05)
    history = _turns(5) + [HumanMessage(content="and my wife?", id="hx"),
                           AIMessage(content="how old is she?", id="rx")]
    assert len(history) == 22
    seen: dict = {}

    class _Capture:
        def invoke(self, messages):
            seen["messages"] = messages
            return AIMessage(content="ok", id="final")

    original, agent.llm = agent.llm, _Capture()
    try:
        agent._model({"messages": history, "rounds": 0},
                     {"configurable": {"trace": Trace()}})
    finally:
        agent.llm = original

    sent = seen["messages"]
    call_ids = {tc["id"] for m in sent
                if isinstance(m, AIMessage) for tc in (m.tool_calls or [])}
    orphans = [m.tool_call_id for m in sent
               if isinstance(m, ToolMessage) and m.tool_call_id not in call_ids]
    assert orphans == [], f"tool results with no call: {orphans}"


def test_a_blocked_answer_is_not_replayed_as_the_models_own_words():
    """Observed live: one blocked turn poisoned the whole thread.

    Bedrock returns its block message as the assistant's content. Stored
    unmarked, it came back on the next turn as the model's own prior answer -
    so the model repeated "I could not give you a reliable answer" with
    nothing blocking it, simply following its own apparent precedent. The
    customer still sees it; the model must not learn from it.
    """
    from app.agents.graph import model_visible

    agent = agent_for(BOT_05)

    class _Blocked:
        def invoke(self, _messages):
            return AIMessage(
                content="I could not give you a reliable answer to that.",
                id="blocked",
                response_metadata={"stopReason": "guardrail_intervened"})

    original, agent.llm = agent.llm, _Blocked()
    try:
        out = agent._model({"messages": [HumanMessage(content="hi", id="h")],
                            "rounds": 0},
                           {"configurable": {"trace": Trace()}})
    finally:
        agent.llm = original

    blocked = out["messages"][0]
    assert blocked.additional_kwargs.get("system_authored") is True
    assert model_visible([blocked]) == []


def test_a_block_message_already_in_a_thread_is_healed_not_just_prevented(monkeypatch):
    """The fault this actually presented as.

    Marking the message when the guardrail fires only covers the first one.
    Bedrock returns the block message as the assistant's content, so once one
    sits in the history unmarked the model writes it again as its own answer -
    and that copy arrives with NO intervention to detect, so it is stored
    unmarked too. The thread then refuses everything with an empty guardrail
    trace beside it, which is how it looked on a live thread: four unmarked
    copies, all visible to the model.

    Matching the configured wording heals a thread already carrying them.
    """
    from app.agents import graph as G

    message = ("I could not give you a reliable answer to that, so I would "
               "rather not guess. Shall I put you through to a colleague who "
               "can check it properly?")
    monkeypatch.setattr(G, "settings", lambda: type(
        "C", (), {"guardrail_blocked_message": message})())

    history = [
        HumanMessage(content="what is the maternity waiting period?", id="h1"),
        AIMessage(content=message, id="a1"),          # no marker: the copy
        HumanMessage(content="and for my wife?", id="h2"),
        AIMessage(content="  I COULD not give you a reliable answer to that, "
                          "so I would rather not guess. Shall I put you "
                          "through to a colleague who can check it properly? ",
                  id="a2"),                            # folded/cased variant
        AIMessage(content="Maternity is covered after 36 months [1].", id="a3"),
    ]

    visible = G.model_visible(history)
    assert [m.id for m in visible] == ["h1", "h2", "a3"]


def test_an_ordinary_answer_is_not_mistaken_for_the_block_message(monkeypatch):
    from app.agents import graph as G

    monkeypatch.setattr(G, "settings", lambda: type(
        "C", (), {"guardrail_blocked_message": "I could not give you a "
                                               "reliable answer to that."})())
    keep = AIMessage(content="I could not find that in the policy wording, "
                             "but I can check with a colleague.", id="a1")
    assert G.model_visible([keep]) == [keep]


def test_with_no_guardrail_deployed_nothing_is_filtered_on_text(monkeypatch):
    """Offline and in tests the message is empty, and an empty string must not
    match every message."""
    from app.agents import graph as G

    monkeypatch.setattr(G, "settings", lambda: type(
        "C", (), {"guardrail_blocked_message": ""})())
    msgs = [AIMessage(content="", id="a1"), AIMessage(content="hello", id="a2")]
    assert G.model_visible(msgs) == msgs
