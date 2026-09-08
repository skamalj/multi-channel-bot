"""The two checkpointers, and what must never end up in one.

ME-1 keys the resolver thread by PERSON; ME-2 keys the bot thread by person
AND line of business. LangGraph owns persistence on both - these tests are
about the boundary between what is state and what is one turn's context.
"""
from __future__ import annotations

from app.agents.graph import agent_for
from app.agents.registry import BOT_02, BOT_05, BOT_06
from app.orchestrator import bot_thread
from app.resolver.graph import resolver_graph, thread_for

CUSTOMER = "919820000009"


def _bot_values(spec, user_id, lob):
    return dict(agent_for(spec).graph.get_state(
        {"configurable": {"thread_id": bot_thread(user_id, lob)}}).values or {})


def _say(client, user, text, **kw):
    r = client.post("/api/chat", json={"user_id": user, "text": text, **kw})
    assert r.status_code == 200, r.text
    return r.json()


# --- the threads -----------------------------------------------------------
def test_the_two_threads_are_keyed_as_documented(client):
    """resolver = user; bot = user#lob."""
    assert thread_for(CUSTOMER) == CUSTOMER
    assert bot_thread(CUSTOMER, "health") == f"{CUSTOMER}#health"
    assert bot_thread(CUSTOMER, "motor") == f"{CUSTOMER}#motor"


def test_a_graph_is_compiled_with_a_checkpointer(client):
    """Not an assertion about style: without one, nothing persists between
    turns and every conversation starts from nothing."""
    assert agent_for(BOT_05).graph.checkpointer is not None
    assert resolver_graph().checkpointer is not None


def test_each_line_of_business_gets_its_own_thread(client):
    """ME-2 through the real API: a health turn and a motor turn leave two
    separate checkpoints, and neither can address the other."""
    _say(client, CUSTOMER, "what is the waiting period for pre-existing disease")
    _say(client, CUSTOMER, "actually my car renewal is due")

    health = _bot_values(BOT_05, CUSTOMER, "health")
    motor = _bot_values(BOT_06, CUSTOMER, "motor")
    assert health.get("lob") == "health" and motor.get("lob") == "motor"

    health_text = " ".join(str(m.content) for m in health["messages"])
    assert "car renewal" not in health_text


def test_the_resolver_thread_survives_across_lines_of_business(client):
    """ME-1. The ledger is about the person, so it spans both compartments -
    which is exactly why it is a different thread with a different TTL."""
    _say(client, CUSTOMER, "what is the waiting period for pre-existing disease")
    _say(client, CUSTOMER, "actually my car renewal is due")

    from app.resolver.graph import hydrate

    values = resolver_graph().get_state(
        {"configurable": {"thread_id": thread_for(CUSTOMER)}}).values
    session = hydrate(values["session"], CUSTOMER)
    lobs = [e.lob for e in session.ledger.events]
    assert "health" in lobs and "motor" in lobs


# --- what must NOT be checkpointed ----------------------------------------
def test_the_per_turn_context_is_never_checkpointed(client):
    """The expensive one.

    `ctx` carries the idempotency key, the consent snapshot and the caller's
    identity - all true for exactly one turn. Checkpointed, they come back on
    the next turn and the double-write protection inverts into a double-write
    cause. They travel in `configurable`, which LangGraph does not persist.
    """
    _say(client, CUSTOMER, "what is the waiting period for pre-existing disease")
    values = _bot_values(BOT_05, CUSTOMER, "health")

    for leaked in ("ctx", "trace", "idempotency_key", "consent",
                   "authenticated", "producer_id"):
        assert leaked not in values, f"{leaked} was checkpointed"


def test_per_turn_scratch_is_reset_on_the_way_in(client):
    """Retrieval results and tool names are per-turn. They live in state
    because everything in a graph does, so the entry node clears them."""
    from app.agents.graph import MAX_TOOL_ROUNDS

    _say(client, CUSTOMER, "what is the waiting period for pre-existing disease")
    after_retrieval = _bot_values(BOT_05, CUSTOMER, "health")
    assert after_retrieval.get("retrieved")
    first_rounds = after_retrieval.get("rounds", 0)

    # A turn that retrieves nothing must not inherit the last turn's chunks.
    _say(client, CUSTOMER, "thanks")
    assert not _bot_values(BOT_05, CUSTOMER, "health").get("retrieved")

    # And the round counter counts THIS turn, not the conversation. Left
    # accumulating, every conversation would hit the tool-round limit and
    # hand off after a few turns.
    for _ in range(3):
        _say(client, CUSTOMER, "what is the initial waiting period")
    assert _bot_values(BOT_05, CUSTOMER, "health")["rounds"] == first_rounds
    assert first_rounds <= MAX_TOOL_ROUNDS


# --- journey state DOES persist -------------------------------------------
def test_a_parked_confirmation_is_checkpointed_not_parked_in_memory(client):
    """AG-6 with no interrupts: waiting is a field that is still empty, and
    it survives because the checkpointer wrote it - not because a process
    stayed alive."""
    client.post("/api/consent", json={"user_id": CUSTOMER,
                                      "purpose": "quotation", "granted": True})
    _say(client, CUSTOMER, "please give me a health quote")

    values = _bot_values(BOT_05, CUSTOMER, "health")
    pending = values.get("pending_confirmation")
    assert pending and pending["tool"] == "quote_create_health"
    assert pending["token"]

    done = _say(client, CUSTOMER, "yes")
    assert [e["label"] for e in done["trace"]["events"]
            if e["kind"] == "tool"] == ["quote_create_health"]
    assert not _bot_values(BOT_05, CUSTOMER, "health").get("pending_confirmation")


def test_the_checkpointer_keeps_a_history_we_can_read_back(client):
    """State history comes free with a checkpointer, and it is what makes the
    glass box able to show a turn that has already finished."""
    _say(client, CUSTOMER, "what is the waiting period for pre-existing disease")
    _say(client, CUSTOMER, "and what about maternity")

    history = list(agent_for(BOT_05).graph.get_state_history(
        {"configurable": {"thread_id": bot_thread(CUSTOMER, "health")}}))
    assert len(history) > 2


def test_erasure_deletes_the_thread_rather_than_blanking_it(client):
    """ME-8. `delete_thread` is the checkpointer's own primitive, so the
    registry can promise erasure rather than approximate it."""
    _say(client, CUSTOMER, "what is the waiting period for pre-existing disease")
    assert _bot_values(BOT_05, CUSTOMER, "health").get("messages")

    report = client.post(f"/api/erase/{CUSTOMER}").json()["report"]
    assert report["bot_session"] >= 1
    assert not _bot_values(BOT_05, CUSTOMER, "health").get("messages")


# --- the DynamoDB saver ----------------------------------------------------
class FakeTable:
    """An in-memory stand-in for a DynamoDB table with (pk, sk).

    Enough to exercise the saver's real logic - key construction, serde,
    ordering, pending writes, erasure - without an AWS account. The four
    storage primitives are the only surface it has to satisfy.
    """

    def __init__(self):
        self.rows: dict[tuple[str, str], dict] = {}

    def put(self, item):
        self.rows[(item["pk"], item["sk"])] = dict(item)

    def get(self, pk, sk):
        return self.rows.get((pk, sk))

    #: Rows returned per "page". DynamoDB caps a query response at 1 MB;
    #: the double caps it at a row count so the same truncation is
    #: reproducible without writing a megabyte.
    page_size = 3

    def query_page(self, pk, prefix, forward=True, limit=None, start=None):
        hits = sorted((k[1], v) for k, v in self.rows.items()
                      if k[0] == pk and k[1].startswith(prefix))
        if not forward:
            hits.reverse()
        if start is not None:
            hits = [h for h in hits if (h[0] > start if forward
                                        else h[0] < start)]
        page = hits[:self.page_size]
        nxt = page[-1][0] if len(hits) > self.page_size else None
        return [v for _, v in page], nxt

    def query(self, pk, prefix, forward=True, limit=None):
        """What a NON-paginating caller would get - the first page only."""
        return self.query_page(pk, prefix, forward, limit)[0]

    def delete(self, keys):
        for k in keys:
            self.rows.pop(k, None)


def _fake_saver(ttl_days=30):
    """A real DynamoDBSaver with its storage primitives redirected."""
    from app.memory.checkpoint import DynamoDBSaver

    saver = DynamoDBSaver.__new__(DynamoDBSaver)   # skip the boto3 __init__
    from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

    BaseInit = type(saver).__mro__[1]
    BaseInit.__init__(saver, serde=JsonPlusSerializer())
    saver.table_name, saver.ttl_seconds = "fake", ttl_days * 86400
    table = FakeTable()
    saver._put = table.put
    saver._get = table.get

    def paginated(pk, prefix, forward=True, limit=None):
        """What the real `_query_prefix` does: follow LastEvaluatedKey.

        The double pages by row count rather than by 1 MB, so the truncation
        DynamoDB produces is reproducible without writing a megabyte.
        """
        items, start = [], None
        while True:
            page, start = table.query_page(pk, prefix, forward, limit, start)
            items.extend(page)
            if not start or (limit and len(items) >= limit):
                break
        return items[:limit] if limit else items

    saver._query_prefix = paginated
    saver._delete_many = table.delete
    saver._b = lambda v: v                        # no boto3 Binary wrapper
    return saver, table


def test_the_dynamodb_saver_round_trips_a_checkpoint():
    """The saver had never executed a single statement. This runs its real
    put/get/list/erase logic against an in-memory table."""
    from langgraph.checkpoint.base import empty_checkpoint

    saver, table = _fake_saver()
    cfg = {"configurable": {"thread_id": "u#health", "checkpoint_ns": ""}}

    cp = empty_checkpoint()
    saved = saver.put(cfg, cp, {"source": "input", "step": 1}, {})
    assert saved["configurable"]["checkpoint_id"] == cp["id"]

    got = saver.get_tuple(cfg)
    assert got is not None
    assert got.checkpoint["id"] == cp["id"]
    assert got.metadata["step"] == 1

    # Every row carries the TTL the store was configured with.
    assert all("ttl" in row for row in table.rows.values())


def test_the_saver_returns_the_latest_checkpoint_and_lists_history():
    from langgraph.checkpoint.base import empty_checkpoint

    saver, _ = _fake_saver()
    cfg = {"configurable": {"thread_id": "u#health", "checkpoint_ns": ""}}

    ids = []
    for step in range(3):
        cp = empty_checkpoint()
        cp["id"] = f"0000000{step}-aaaa"
        ids.append(cp["id"])
        saver.put(cfg, cp, {"source": "loop", "step": step}, {})

    assert saver.get_tuple(cfg).checkpoint["id"] == ids[-1]
    assert [t.checkpoint["id"] for t in saver.list(cfg)] == list(reversed(ids))


def test_the_saver_keeps_pending_writes_with_their_checkpoint():
    from langgraph.checkpoint.base import empty_checkpoint

    saver, _ = _fake_saver()
    cp = empty_checkpoint()
    cfg = {"configurable": {"thread_id": "u#health", "checkpoint_ns": ""}}
    saver.put(cfg, cp, {"source": "input", "step": 1}, {})

    at = {"configurable": {"thread_id": "u#health", "checkpoint_ns": "",
                           "checkpoint_id": cp["id"]}}
    saver.put_writes(at, [("messages", "hello"), ("slots", {"idv": 480000})],
                     task_id="task-1")

    writes = saver.get_tuple(cfg).pending_writes
    assert ("task-1", "messages", "hello") in writes
    assert ("task-1", "slots", {"idv": 480000}) in writes


def test_deleting_a_thread_removes_its_checkpoints_and_writes():
    """ME-8: erasure is the checkpointer's own primitive, not an empty state
    written over the top."""
    from langgraph.checkpoint.base import empty_checkpoint

    saver, table = _fake_saver()
    cfg = {"configurable": {"thread_id": "u#health", "checkpoint_ns": ""}}
    cp = empty_checkpoint()
    saver.put(cfg, cp, {"source": "input", "step": 1}, {})
    saver.put_writes({"configurable": dict(cfg["configurable"],
                                           checkpoint_id=cp["id"])},
                     [("messages", "hello")], task_id="t1")
    # A second thread must survive - erasure is per person, not per table.
    saver.put({"configurable": {"thread_id": "other#motor",
                                "checkpoint_ns": ""}},
              empty_checkpoint(), {"source": "input", "step": 1}, {})

    assert len(table.rows) >= 3
    saver.delete_thread("u#health")
    assert saver.get_tuple(cfg) is None
    assert any(pk == "other#motor" for pk, _ in table.rows)


def test_the_two_savers_differ_only_in_table_and_ttl():
    """The separation is a fact about addressing and retention, not
    behaviour - so the code path is identical."""
    from app.config import settings

    cfg = settings()
    assert cfg.session_ttl_days != cfg.resolver_ttl_days
    assert cfg.ddb_checkpoint_table != cfg.ddb_resolver_table


def test_no_application_class_is_ever_put_in_a_checkpoint(client):
    """LangGraph warns on deserialising unregistered types and will block it
    in a future version - rightly, because a checkpoint that can rehydrate
    arbitrary classes is a deserialisation gadget. Its permissive default
    cannot be added to, so the durable answer is to persist DATA: the
    resolver thread holds a dict, not a `ResolverSession`."""
    _say(client, CUSTOMER, "what is the waiting period for pre-existing disease")

    values = resolver_graph().get_state(
        {"configurable": {"thread_id": thread_for(CUSTOMER)}}).values
    assert isinstance(values["session"], dict),         "the resolver thread persisted a class, not data"
    assert values["session"]["user_id"] == CUSTOMER
    assert isinstance(values["session"]["ledger"], dict)


def test_the_session_round_trips_as_data(client):
    """Dumped in, validated out - and the ledger survives intact."""
    from app.resolver.graph import hydrate
    from app.resolver.ledger import RouteEvent
    from app.resolver.store import ResolverSession

    session = ResolverSession(user_id="u1", persona="customer", turn=3)
    session.ledger.append(RouteEvent(turn=1, persona="customer", lob="health",
                                     decided_by="explicit", confidence=0.95))
    back = hydrate(session.model_dump(), "u1")

    assert isinstance(back, ResolverSession)
    assert back.turn == 3 and back.ledger.events[0].lob == "health"


def test_a_query_that_does_not_paginate_truncates_silently():
    """The bug real DynamoDB found and the in-memory double had hidden.

    A query response is capped at 1 MB. Checkpoints hold whole message
    lists, so a busy thread passes that quickly - and an unpaginated read
    reports a thread erased while rows remain, and shows a partial history
    as if it were the whole one.
    """
    from langgraph.checkpoint.base import empty_checkpoint

    saver, table = _fake_saver()
    cfg = {"configurable": {"thread_id": "u#health", "checkpoint_ns": ""}}
    for i in range(table.page_size * 3):
        cp = empty_checkpoint()
        cp["id"] = f"0000{i:04d}-aaaa"
        saver.put(cfg, cp, {"source": "loop", "step": i}, {})

    # The paginating read sees everything; one page would have seen three.
    assert len(saver._query_prefix("u#health", "cp#")) == table.page_size * 3
    assert len(table.query("u#health", "cp#")) == table.page_size

    # And erasure loops until the thread is actually empty.
    saver.delete_thread("u#health")
    assert saver.get_tuple(cfg) is None
    assert not table.rows


def test_reads_are_strongly_consistent():
    """DynamoDB reads are eventually consistent by default. A checkpointer
    that reads a stale checkpoint loses the previous turn: the customer says
    "yes" and the confirmation it was answering is not there yet."""
    import inspect

    from app.memory.checkpoint import DynamoDBSaver

    for fn in (DynamoDBSaver._get, DynamoDBSaver._query_prefix):
        assert "ConsistentRead" in inspect.getsource(fn), fn.__name__


def test_a_parked_confirmation_does_not_swallow_the_conversation():
    """Found by replaying a real transcript against the deployed runtime.

    While a confirmation was parked, every message was read ONLY as an answer
    to it. "What is copayment?" came back as "I am about to create a health
    quote - shall I go ahead?", and so did the question after that. The
    customer could not change the subject until they said yes, said no, or
    waited out the 30-minute expiry.

    An unclear answer now carries on to the model, and the confirmation stays
    parked so yes still works afterwards.
    """
    from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

    from app.agents.graph import agent_for
    from app.agents.registry import BOT_05

    agent = agent_for(BOT_05)

    # The customer asked something else entirely.
    asked_something_else = {"messages": [HumanMessage(content="what is "
                                                              "copayment?")]}
    assert agent._after_pending(asked_something_else) == "model"

    # They said yes, the tool ran, and the model speaks to the result.
    confirmed = {"messages": [ToolMessage(content="{}", tool_call_id="t")]}
    assert agent._after_pending(confirmed) == "model"

    # They declined; that is already answered and the turn is over.
    declined = {"messages": [AIMessage(content="No problem - I have not "
                                               "done that.")]}
    assert agent._after_pending(declined) == "end"
