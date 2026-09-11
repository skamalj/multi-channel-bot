"""The bot: a react agent per configuration.

`create_agent` owns the turn - the model<->tools loop, the message list, tool
execution and the AI/Tool pairing Bedrock requires. We add four things, all as
standard middleware, and nothing else:

    InputGuardrail     before_agent   model-based input screen (short-circuits)
    Reducer            before_model   our history reducer (state management)
    ToolControls       wrap_tool_call authorization / consent / hitl
    ModelCallLimit     built-in       bound the loop (was our round counter)
    ToolCallLimit      built-in
    GroundingGuardrail after_agent    verify citations, drop/refuse ungrounded

Workflow lives in the prompt (`prompts/*.md`), never in graph branches.
Guardrails are provider-agnostic MIDDLEWARE (app/agents/guardrails.py,
grounding.py), not a property of the model object - `get_llm()` returns a plain
model with no vendor guardrail attached. Identity and consent arrive per turn
as `RequestContext` and are never in the model's schema or the checkpoint.

An agent IS its tag set, bound once here, which is what keeps the capability
matrix printable without simulating a conversation.
"""
from __future__ import annotations

from langchain.agents import create_agent
from langchain.agents.middleware import (ModelCallLimitMiddleware,
                                         ToolCallLimitMiddleware)

from app.agents.context import RequestContext
from app.agents.controls import ToolControls
from app.agents.grounding import GroundingGuardrail
from app.agents.guardrails import InputGuardrail
from app.agents.reduce import Reducer
from app.agents.state import BotState
from app.config import settings
from app.llm.bedrock import get_llm
from app.mcpserver.registry import tools_for
from app.memory.checkpoint import bot_checkpointer
from app.resolver.spec import AgentSpec


def _system_prompt(spec: AgentSpec) -> str:
    """The prompt files carry the workflow. Here we append only the binding
    facts - the line of business and corpus scope the tools were actually
    bound with, and the versions that must travel in the trace. These come
    from the binding, not the author, so they cannot live in the file."""
    cfg = settings()
    return (
        f"{spec.system_prompt}\n\n"
        f"Active line of business: {spec.lob}. You have no access to any "
        f"other line of business and must not speculate about one.\n"
        f"Retrieval scope: {', '.join(spec.corpus_scope)}.\n"
        f"prompt_version={cfg.prompt_version} config_version={cfg.config_version}"
    )


class Agent:
    """A compiled react agent plus the manifest facts the orchestrator traces.

    `.graph` is the compiled agent - it has the full `invoke`/`get_state`/
    `update_state`/`get_state_history` surface the orchestrator uses, because
    `create_agent` returns a `CompiledStateGraph` like any other.
    """

    def __init__(self, spec: AgentSpec):
        self.spec = spec
        self.tools = tools_for(spec.tool_tags)
        self.tool_names = sorted(t.name for t in self.tools)
        cfg = settings()
        self.graph = create_agent(
            model=get_llm(),
            tools=self.tools,
            system_prompt=_system_prompt(spec),
            middleware=[
                InputGuardrail(),
                Reducer(),
                ToolControls(),
                ModelCallLimitMiddleware(run_limit=cfg.model_call_limit,
                                         exit_behavior="end"),
                ToolCallLimitMiddleware(run_limit=cfg.tool_call_limit,
                                        exit_behavior="end"),
                GroundingGuardrail(),
            ],
            state_schema=BotState,
            context_schema=RequestContext,
            checkpointer=bot_checkpointer(),
        )


_AGENTS: dict[str, Agent] = {}


def agent_for(spec: AgentSpec) -> Agent:
    if spec.key not in _AGENTS:
        _AGENTS[spec.key] = Agent(spec)
    return _AGENTS[spec.key]


def reset_agents() -> None:
    """Tests switch MOCK_LLM and checkpointers between cases."""
    _AGENTS.clear()
