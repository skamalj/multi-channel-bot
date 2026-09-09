"""The two things that replaced hand-written flow control.

Instructions live in `prompts/` as files somebody can read and diff, and the
agent loop is bounded by LangGraph's own `recursion_limit` rather than by a
counter that routed past the tools node to enforce itself.
"""
from __future__ import annotations

import pytest


# --- prompts ---------------------------------------------------------------
def test_every_configuration_loads_its_instructions_from_a_file():
    from app.agents.registry import REGISTRY

    for spec in REGISTRY.values():
        assert spec.system_prompt.strip(), f"{spec.bot_id} has no prompt"
        # The common file is prepended to every one of them.
        assert "Protec assistant" in spec.system_prompt, spec.bot_id


def test_the_workflow_the_graph_used_to_hardcode_is_in_the_prompt():
    """These are the sentences the graph used to compose and inject: asking
    before a write, saying plainly when something was declined, and what to
    do when a tool refuses for want of consent."""
    from app.agents.prompts import load

    text = load("health_customer.md").lower()
    for instruction in ("before calling one of those",
                        "wait", "not agreement", "decline",
                        "consent_required", "consent_grant"):
        assert instruction in text, f"the prompt never mentions {instruction!r}"


def test_a_missing_prompt_file_fails_loudly_rather_than_running_without_it():
    """A bot silently running with half its instructions is worse than one
    that will not start - the missing half is invariably the part that says
    what not to do."""
    from app.agents.prompts import PromptMissing, load

    with pytest.raises(PromptMissing):
        load("no_such_persona.md")


# --- the loop --------------------------------------------------------------
def test_nothing_counts_rounds_to_bound_the_loop():
    """The counter is gone, and with it the branch that enforced it by
    routing PAST the tools node - which left the model's calls unexecuted
    and unanswered, a thread Bedrock rejects on the next turn."""
    from app.agents import graph as G

    assert not hasattr(G, "MAX_TOOL_ROUNDS")
    assert not hasattr(G.Agent, "_after_tools")
    assert not hasattr(G.Agent, "_final")


def test_a_tool_call_always_reaches_the_tools_node():
    """The invariant that makes an orphaned tool call impossible."""
    from langchain_core.messages import AIMessage

    from app.agents.graph import agent_for
    from app.agents.registry import BOT_05

    agent = agent_for(BOT_05)
    calling = {"messages": [AIMessage(content="", id="a1", tool_calls=[
        {"name": "kb_search_health", "args": {"query": "x"}, "id": "tc1"}])]}
    assert agent._after_model(calling) == "tools"

    answering = {"messages": [AIMessage(content="here you go", id="a2")]}
    assert agent._after_model(answering) == "respond"


def test_the_graph_is_bounded_by_the_frameworks_limit():
    """A model that will not stop calling tools is stopped by LangGraph, not
    by us. GraphRecursionError is the framework saying the loop will not
    settle; the orchestrator turns it into an honest reply."""
    from langgraph.errors import GraphRecursionError

    from app.agents.graph import agent_for
    from app.agents.registry import BOT_05

    class _Relentless:
        """Always asks for another tool. Never answers."""

        def bind_tools(self, _tools):
            return self

        def invoke(self, _messages):
            from langchain_core.messages import AIMessage
            _Relentless.n = getattr(_Relentless, "n", 0) + 1
            return AIMessage(content="", id=f"a{_Relentless.n}", tool_calls=[
                {"name": "kb_search_health", "args": {"query": "again"},
                 "id": f"tc{_Relentless.n}"}])

    agent = agent_for(BOT_05)
    original, agent.llm = agent.llm, _Relentless()
    try:
        with pytest.raises(GraphRecursionError):
            agent.graph.invoke(
                {"messages": [], "user_id": "u-recursion",
                 "bot_id": "BOT-05", "persona": "customer", "lob": "health"},
                {"configurable": {"thread_id": "recursion-test"},
                 "recursion_limit": 8})
    finally:
        agent.llm = original
