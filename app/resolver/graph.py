"""The resolver graph.

Two nodes, one checkpointer, `thread_id = user` (ME-1). It holds the route
ledger, the shared profile and auth state - everything that is true about a
PERSON rather than about one line of business - and it is the reason the bot
graph can be written as if each bot were the only bot.

    START ─ persona ─┬─ END        (step-up needed: the turn stops here)
                     └─ route ─ END

Why a graph for something deterministic: the checkpointer is what persists
the ledger between turns, and a checkpointer attaches to a compiled graph.
The nodes themselves stay deterministic - regex, thresholds, and one small
model call that fails closed - because routing decisions have to be
explainable months later.

Note the two TTLs this creates: 180 days here, 30 on the bot thread. The
ledger improves with age; the bot session holds the most sensitive data and
should not.
"""
from __future__ import annotations

from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from app.memory.checkpoint import resolver_checkpointer
from app.obs.trace import Trace
from app.resolver.resolver import refresh_holdings, resolve, resolve_persona
from app.resolver.store import ResolverSession


class ResolverState(TypedDict, total=False):
    # Persisted: the whole point of this thread - and persisted as DATA.
    # A checkpoint holding our own classes is a class the serializer has to
    # be told to trust, which is a dependency bump away from breaking and a
    # deserialisation gadget in the meantime.
    session: dict

    # Per-turn inputs, supplied on every invocation.
    text: str | None
    entry_persona: str | None
    entry_lob: str | None
    work_in_flight: bool
    display_name: str | None
    channel: str | None

    # Per-turn outputs, read back by the orchestrator.
    persona: str | None
    lob: str | None
    question: str | None


def _trace(config) -> Trace:
    return (config or {}).get("configurable", {}).get("trace") or Trace()


def hydrate(session: dict | None, user_id: str = "") -> ResolverSession:
    """The checkpoint holds a dict; the nodes work with the model."""
    if not session:
        return ResolverSession(user_id=user_id)
    return ResolverSession.model_validate(session)


def _hydrate(state: ResolverState) -> ResolverSession:
    return hydrate(state.get("session"), state.get("user_id", ""))


def _persona(state: ResolverState, config) -> dict:
    """Persona from the producer directory, every turn.

    Never inferred from history - otherwise a long enough conversation
    becomes a privilege-escalation path.
    """
    trace = _trace(config)
    session = _hydrate(state)
    session.turn += 1

    from app.memory import profile as profile_mem

    if state.get("display_name") and not session.profile.display_name:
        profile_mem.update(session, display_name=state["display_name"])
    if state.get("channel") and session.profile.channel != state["channel"]:
        profile_mem.update(session, channel=state["channel"])

    persona, _source, question = resolve_persona(
        session.user_id, session, state.get("entry_persona"), trace)
    return {"session": session.model_dump(), "persona": persona,
            "question": question}


def _after_persona(state: ResolverState) -> str:
    # RS-9: a step-up is needed, so the turn stops before any LOB work.
    return "end" if state.get("question") else "route"


def _route(state: ResolverState, config) -> dict:
    """RS-3's ladder. Deterministic signals first, the model last, and a
    question before a guess."""
    trace = _trace(config)
    session = _hydrate(state)

    # Holdings are facts ABOUT a line of business, read from the policy
    # directory - which is what makes them shareable across a compartment.
    refresh_holdings(session.user_id, session, trace)

    spec, question = resolve(state.get("text"), state.get("entry_lob"),
                             session, state["persona"],
                             bool(state.get("work_in_flight")), trace)
    return {"session": session.model_dump(), "question": question,
            "lob": spec.lob if spec else None}


def _build():
    g = StateGraph(ResolverState)
    g.add_node("persona", _persona)
    g.add_node("route", _route)
    g.add_edge(START, "persona")
    g.add_conditional_edges("persona", _after_persona,
                            {"route": "route", "end": END})
    g.add_edge("route", END)
    return g.compile(checkpointer=resolver_checkpointer())


_GRAPH: Any = None


def resolver_graph():
    global _GRAPH
    if _GRAPH is None:
        _GRAPH = _build()
    return _GRAPH


def reset_resolver_graph() -> None:
    """Tests rebuild against a fresh checkpointer."""
    global _GRAPH
    _GRAPH = None


def thread_for(user_id: str) -> str:
    """ME-1: the resolver thread is the PERSON, not a configuration."""
    return user_id
