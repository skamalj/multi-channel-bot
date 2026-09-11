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


def test_the_loop_is_bounded_by_the_frameworks_limit_middleware():
    """A model that will not stop calling tools is stopped by the framework's
    ToolCallLimit middleware, not by a counter of ours. exit_behavior="end"
    ends the turn cleanly rather than raising, so the run terminates instead
    of looping forever."""
    from langchain.agents import create_agent
    from langchain.agents.middleware import ToolCallLimitMiddleware
    from langchain_core.messages import AIMessage, HumanMessage
    from langchain_core.tools import tool as make_tool

    calls = {"n": 0}

    @make_tool
    def ping() -> str:
        """ping"""
        return "pong"

    class Relentless:
        """Always asks for another tool. Never answers."""

        def bind_tools(self, tools, **_):
            return self

        def invoke(self, messages, **_):
            calls["n"] += 1
            return AIMessage(content="", id=f"a{calls['n']}", tool_calls=[
                {"name": "ping", "args": {}, "id": f"tc{calls['n']}"}])

    agent = create_agent(
        model=Relentless(), tools=[ping],
        middleware=[ToolCallLimitMiddleware(run_limit=3, exit_behavior="end")])
    out = agent.invoke({"messages": [HumanMessage(content="go")]})

    # It terminated (did not loop forever) and stopped at the limit.
    assert calls["n"] <= 4, f"the limit did not stop the loop: {calls['n']}"
    assert out["messages"], "the run produced no messages"


def test_the_idempotency_key_is_the_call_not_the_turn():
    """A quote agreed to twice is one quote.

    The key is derived in the write tool from its own arguments, so it is
    identical across turns and retries (same call, same key) and differs when
    the arguments do (a different operation). The store replays the first
    result on a repeat rather than writing again.
    """
    from app.agents import confirm
    from app.coremock import rating, store

    args = {"product_id": "PHS", "sum_insured": 500000,
            "member_ages": [40], "city": "Pune", "addons": None}
    key = confirm.token_for("quote_create_health", args)

    # Same call -> same key, every turn. Different args -> different operation.
    assert confirm.token_for("quote_create_health", args) == key
    assert confirm.token_for(
        "quote_create_health", {**args, "sum_insured": 1000000}) != key

    # And the store replays the first result rather than writing a second.
    store.reset_chaos()
    first = store.save_quote(rating.rate_health("PHS", 500000, [40], "Pune", []),
                             idempotency_key=key)
    again = store.save_quote(rating.rate_health("PHS", 500000, [40], "Pune", []),
                             idempotency_key=key)
    assert again.get("idempotent_replay")
    assert again["quote_id"] == first["quote_id"]
