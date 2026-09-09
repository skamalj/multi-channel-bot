"""The seven bots are a two-dimensional lookup, not seven applications.

Two configurations are built here - BOT-05 health/customer and BOT-02
motor/agent - plus the customer-side motor bot the demo needs so that a
mid-conversation line-of-business switch has somewhere to land. Adding the
rest is adding rows, not applications: a bot IS a persona, a line of
business, a tag set and a prompt.

The prompts are files under `prompts/`, not strings built here - see
app/agents/prompts.py.
"""
from __future__ import annotations

from app.agents.prompts import load as load_prompt
from app.resolver.spec import AgentSpec

BOT_05 = AgentSpec(
    bot_id="BOT-05",
    name="health_customer",
    persona="customer",
    lob="health",
    tool_tags={"lob": "health", "persona": "customer"},
    corpus_scope=["public"],
    greeting="Hi! I can help you find health cover, understand your policy, "
             "or raise a claim. What would you like to do?",
    system_prompt=load_prompt("health_customer.md"),
)

BOT_02 = AgentSpec(
    bot_id="BOT-02",
    name="motor_agent",
    persona="agent",
    lob="motor",
    tool_tags={"lob": "motor", "persona": "agent"},
    corpus_scope=["public", "agent"],
    greeting="Hi! Vehicle number to start a quote, or ask me anything about "
             "the motor book.",
    system_prompt=load_prompt("motor_agent.md"),
)

BOT_06 = AgentSpec(
    bot_id="BOT-06",
    name="motor_customer",
    persona="customer",
    lob="motor",
    tool_tags={"lob": "motor", "persona": "customer"},
    corpus_scope=["public"],
    greeting="Hi! I can help with your car or two-wheeler policy - renewal, "
             "a claim, or finding a garage.",
    system_prompt=load_prompt("motor_customer.md"),
)

REGISTRY: dict[str, AgentSpec] = {s.key: s for s in (BOT_05, BOT_02, BOT_06)}


def spec_for(persona: str, lob: str) -> AgentSpec | None:
    return REGISTRY.get(f"{persona}#{lob}")


def capability_matrix() -> dict[str, dict]:
    """Static, printable: exactly what each bot can ever call.

    Only possible because tool binding happens at build time. An agent whose
    tool set is computed per turn cannot produce this document at all - which
    is why this is the artefact a maker-checker control can actually review.
    """
    from app.mcpserver.registry import list_tools

    out: dict[str, dict] = {}
    for spec in REGISTRY.values():
        tools = sorted(list_tools(match=spec.tool_tags), key=lambda t: t.name)
        out[spec.bot_id] = {
            "name": spec.name,
            "persona": spec.persona,
            "lob": spec.lob,
            "corpus_scope": spec.corpus_scope,
            "tools": [
                {"name": t.name, "effect": t.effect, "authority": t.authority,
                 "auth": t.auth, "pii": t.pii, "confirm": t.confirm,
                 "subject": t.subject, "consent_purpose": t.consent_purpose}
                for t in tools
            ],
        }
    return out
