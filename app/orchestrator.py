"""One turn, end to end.

    channel -> resolver graph (thread = user)
            -> bot graph      (thread = user#lob)
            -> audit          (person, spanning every LOB)

**Two checkpointers, keyed apart** (ME-1, ME-2). LangGraph owns persistence
on both: nothing here serialises a message, computes a TTL or writes a
session row. The orchestrator's whole job is to choose the two thread ids and
invoke in the right order.

The order is deliberate: the bot graph FIRST, the resolver thread SECOND.
They are not atomic, and a ledger that is one turn stale self-corrects on the
next message, whereas a ledger pointing at a session that never advanced
strands the customer (ME-4). Two separate invocations preserve that, because
each checkpoint is written when its own invocation returns.

Per-turn context - the request identity, consent snapshot, idempotency key
and the trace - travels in `configurable` and is never checkpointed.
"""
from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage

from app.agents.graph import agent_for
from app.agents.registry import spec_for
from app.channels.base import IngestEvent, OutboundMessage
from app.config import settings
from app.coremock import store as core_store
from app.memory.longterm import consent, interactions, suppression
from app.obs import audit
from app.obs.trace import Trace
from app.resolver.graph import hydrate, resolver_graph, thread_for


def bot_thread(user_id: str, lob: str) -> str:
    """ME-2. The bot thread is a person AND a line of business.

    Holding one thread with LOB compartments inside it makes
    non-contamination depend on every prompt-assembly path filtering
    correctly, forever. Keyed apart, the leak requires deliberately
    constructing another thread's id.
    """
    return f"{user_id}#{lob}"


def handle(event: IngestEvent) -> tuple[list[OutboundMessage], Trace]:
    cfg = settings()
    trace = Trace()
    trace.add("channel", event.channel, message_id=event.message_id,
              kind=event.kind, identity=event.channel_identity,
              prompt_version=cfg.prompt_version,
              config_version=cfg.config_version)

    resolver = resolver_graph()
    r_config = {"configurable": {"thread_id": thread_for(event.user_id),
                                 "trace": trace}}

    # The prior route tells us which bot thread might hold work in flight,
    # which is what raises the bar for leaving it (RS-5).
    prior = hydrate(resolver.get_state(r_config).values.get("session"),
                    event.user_id)
    prior_lob = prior.ledger.prior_lob()
    work_in_flight = False
    if prior_lob:
        prior_state = _bot_state(event.user_id, prior_lob)
        work_in_flight = bool(prior_state.get("slots")
                              or prior_state.get("pending_confirmation"))

    resolved = resolver.invoke({
        "session": prior.model_dump(),
        "text": event.text, "entry_persona": event.entry_persona,
        "entry_lob": event.entry_lob, "work_in_flight": work_in_flight,
        "display_name": event.display_name, "channel": event.channel,
        "question": None, "lob": None,
    }, r_config)

    session = hydrate(resolved.get("session"), event.user_id)
    persona = resolved.get("persona")
    if resolved.get("question"):
        _audit(event, None, resolved.get("lob"), persona, trace,
               resolved["question"])
        return _reply(event, resolved["question"], trace), trace

    spec = spec_for(persona, resolved["lob"]) or spec_for(persona, "health")
    agent = agent_for(spec)
    trace.add("bind", spec.bot_id, persona=spec.persona, lob=spec.lob,
              tags=spec.tool_tags, tools=agent.tool_names,
              corpus_scope=spec.corpus_scope,
              thread=bot_thread(event.user_id, spec.lob))

    customer = core_store.customer_for(event.user_id)
    producer = core_store.producer_for(event.user_id)
    ctx = {
        "user_id": event.user_id,
        "persona": spec.persona,
        "lob": spec.lob,
        # The corpus this configuration may read. Injected into retrieval,
        # never offered to the model as an argument.
        "corpus_scope": spec.corpus_scope,
        "customer_id": (customer or {}).get("customer_id"),
        "producer_id": (producer or {}).get("producer_id"),
        "authenticated": session.profile.authenticated,
        "consent": {**consent.current(event.user_id),
                    **session.profile.consent},
        "idempotency_key": event.message_id,
    }
    trace.add("bind", "request_context",
              authenticated=ctx["authenticated"],
              consent=sorted(k for k, v in ctx["consent"].items() if v),
              customer_id=ctx["customer_id"], producer_id=ctx["producer_id"])

    b_config = {"configurable": {
        "thread_id": bot_thread(event.user_id, spec.lob),
        "ctx": ctx, "trace": trace}}

    # RS-9: a persona switch carries no journey state. The thread is the
    # person and the line of business, so the switch clears it rather than
    # opening a third compartment per customer.
    if session.persona_switched:
        trace.add("gate", "journey_state_dropped", reason="persona switch")
        _clear_bot_thread(agent, b_config)
        session.persona_switched = False

    # A document is stored and REFERENCED. Its bytes never enter the thread:
    # the customer sees one line, the checkpoint holds metadata, and a tool
    # that needs the content fetches it by reference.
    turn_text = event.text or ""
    new_docs: list[dict] = []
    if event.media is not None:
        ref = _store_document(event, spec.lob, trace)
        if ref is not None:
            new_docs.append(ref.as_dict())
            line = ref.as_message()
            turn_text = f"{turn_text}\n{line}".strip() if turn_text else line

    inputs = {
        "messages": [HumanMessage(content=turn_text)],
        "documents": list(state_documents(agent, event, spec)) + new_docs,
        "user_id": event.user_id, "channel": event.channel,
        "channel_identity": event.channel_identity,
        "last_message_id": event.message_id,
        "bot_id": spec.bot_id, "persona": spec.persona, "lob": spec.lob,
        "corpus_scope": spec.corpus_scope,
        # The ONLY thing that crosses a compartment boundary.
        "shared": session.profile.model_dump(),
    }
    state = agent.graph.invoke(inputs, b_config)   # bot checkpoint FIRST

    # The resolver checkpoint is written by its own invocation above; the
    # ledger update below is the second write, and the stale-by-one-turn
    # case self-corrects.
    resolver.update_state(r_config, {"session": session.model_dump()})

    last = state["messages"][-1]
    text = last.content if isinstance(last, AIMessage) else str(last.content)
    tools = list(state.get("last_tools") or [])
    interactions.record_turn(event.user_id, spec.lob, spec.bot_id, tools,
                             handed_off="human_handoff" in tools)
    _audit(event, spec.bot_id, spec.lob, persona, trace, text,
           list(state.get("last_citations") or []))
    return _reply(event, text or "(no reply)", trace), trace


def state_documents(agent, event: IngestEvent, spec) -> list[dict]:
    """Documents already referenced on this thread."""
    snapshot = agent.graph.get_state(
        {"configurable": {"thread_id": bot_thread(event.user_id, spec.lob)}})
    return list((snapshot.values or {}).get("documents") or [])


def _store_document(event: IngestEvent, lob: str, trace: Trace):
    """Put the bytes in object storage and keep only the reference.

    The content arrives with the channel adapter - a WhatsApp media fetch, a
    console upload. When there is no content yet (the media id is known but
    the fetch is not wired) the metadata is still recorded, because the
    reference is what the rest of the system works with either way.
    """
    from app.storage.documents import documents

    media = event.media
    try:
        ref = documents().put(
            event.user_id, lob, media.filename or media.media_id,
            media.mime, getattr(media, "content", None) or b"")
        trace.add("tool", "document_stored", doc_id=ref.doc_id,
                  mime=ref.mime, size=ref.size, key=ref.key)
        return ref
    except Exception as exc:                                 # noqa: BLE001
        trace.add("error", "document_store_failed",
                  detail=f"{type(exc).__name__}: {exc}")
        return None


def _bot_state(user_id: str, lob: str) -> dict:
    """Read a bot thread without running it - used to see whether work is in
    flight on the route we might be leaving."""
    from app.agents.registry import REGISTRY

    spec = next((s for s in REGISTRY.values() if s.lob == lob), None)
    if spec is None:
        return {}
    snapshot = agent_for(spec).graph.get_state(
        {"configurable": {"thread_id": bot_thread(user_id, lob)}})
    return dict(snapshot.values or {})


def _clear_bot_thread(agent, config) -> None:
    """Drop the journey without dropping the thread.

    The messages stay in the checkpoint history - the audit record is a
    separate concern - but the fields a journey is made of are emptied.
    """
    agent.graph.update_state(config, {
        "slots": {}, "quotes": {}, "pending_confirmation": None,
        "known_facts": [],
    })


def _reply(event: IngestEvent, text: str, trace: Trace) -> list[OutboundMessage]:
    # A suppression is checked before every outbound, on every channel.
    if suppression.suppressed(event.user_id, event.channel):
        trace.add("guardrail", "suppressed", channel=event.channel)
        return []
    return [OutboundMessage(user_id=event.user_id, channel=event.channel,
                            text=text, reply_to=event.message_id)]


def _audit(event: IngestEvent, bot_id, lob, persona, trace: Trace,
           outbound: str, citations: list[str] | None = None) -> None:
    try:
        audit.record(event.user_id, bot_id, lob, persona, trace,
                     event.text, outbound, citations)
    except Exception:                                        # noqa: BLE001
        # An audit failure must be loud in logs and must not lose the reply;
        # in production this queues rather than swallowing.
        trace.add("error", "audit_write_failed")
