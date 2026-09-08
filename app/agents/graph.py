"""The bot graph.

A compiled `StateGraph` per configuration, with the bot checkpointer behind
it. LangGraph owns the thread: `thread_id = user#lob`, and persistence,
message reduction and state history come with it.

    entry ─ reduce ─┬─ pending ─┬─ END          (declined, or asked again)
                    │           └─ model        (confirmed: the tool ran)
                    └─ model ─┬─ tools ── model (up to MAX_TOOL_ROUNDS)
                              └─ respond ── END

**No interrupts anywhere, and that is unchanged by using a graph.** Every
invocation runs start to finish in one turn. A confirmation the customer has
not given yet is a FIELD in the checkpointed state (`pending_confirmation`),
read by the entry node on the next turn - not a parked execution. On Lambda
there is nothing to park, and `interrupt()` would re-run the whole node on
resume, firing any send inside it twice.

Checkpointing and interrupts are independent features. Taking the first and
declining the second is the whole design here.

**Tools are bound once, at graph construction.** An agent IS its tag set,
which is what makes `capability_matrix()` printable without simulating a
conversation.
"""
from __future__ import annotations

import json
import re
from typing import Any

from langchain_core.messages import AIMessage, SystemMessage, ToolMessage
from langgraph.graph import END, START, StateGraph

from app.agents import citations, confirm, guardrail, reduce as reducer
from app.mcpserver.registry import get_tool as spec_for_tool
from app.agents.state import BotState
from app.config import settings
from app.llm.bedrock import get_llm
from app.mcpserver.registry import authorize, get_tool, json_schema, list_tools
from app.memory.checkpoint import bot_checkpointer
from app.obs.trace import Trace
from app.resolver.spec import AgentSpec

MAX_TOOL_ROUNDS = 3
RETRIEVAL_TOOLS = {"kb_search_health", "kb_search_motor"}
KNOWN_FACTS_KEPT = 4

# AG-8: a handoff is the FALLBACK, not the first move. Some models reach for
# it immediately on any question that sounds regulated, which produces a bot
# that queues a colleague for a question the corpus answers in one call. The
# prompt asks for a lookup first; this enforces it, because a control that
# depends on a model reading an instruction is not a control.
#
# The exception is the customer asking for a person in so many words. Making
# somebody argue their way out of a bot is the wrong place to be strict.
WANTS_HUMAN = re.compile(
    r"\b(human|real person|speak to someone|talk to someone|"
    r"representative|customer care|call ?back|call me|escalate|complaint)\b",
    re.I)

HANDOFF = ("I could not complete that in a reasonable number of steps, and I "
           "would rather not keep you guessing. Let me put you through to a "
           "colleague who can pick this up with everything you have told me.")


def _tool_schemas(spec: AgentSpec) -> list[dict]:
    """Bound ONCE per agent, at build time, from the tag manifest."""
    return [
        {"name": t.name, "description": t.description,
         "input_schema": json_schema(t)}
        for t in sorted(list_tools(match=spec.tool_tags), key=lambda s: s.name)
    ]


def _turn(config) -> tuple[dict, Trace]:
    """The per-turn request context, which is NOT checkpointed.

    It travels in `configurable` and is read by the nodes that need it. Put
    any of this in state and it comes back on a later turn - a stale
    idempotency key being the expensive example.
    """
    conf = (config or {}).get("configurable", {})
    return conf.get("ctx") or {}, conf.get("trace") or Trace()


class Agent:
    """A compiled graph plus the manifest facts the orchestrator traces."""

    def __init__(self, spec: AgentSpec):
        self.spec = spec
        self.tool_schemas = _tool_schemas(spec)
        self.tool_names = [s["name"] for s in self.tool_schemas]
        self.llm = get_llm().bind_tools(self.tool_schemas)
        self.graph = self._build()

    # -- graph ------------------------------------------------------------
    def _build(self):
        g = StateGraph(BotState)
        g.add_node("entry", self._entry)
        g.add_node("reduce", self._reduce)
        g.add_node("pending", self._pending)
        g.add_node("model", self._model)
        g.add_node("tools", self._tools)
        g.add_node("respond", self._respond)

        g.add_edge(START, "entry")
        g.add_edge("entry", "reduce")
        g.add_conditional_edges("reduce", self._after_entry,
                                {"pending": "pending", "model": "model"})
        g.add_conditional_edges("pending", self._after_pending,
                                {"model": "model", "end": END})
        g.add_conditional_edges("model", self._after_model,
                                {"tools": "tools", "respond": "respond"})
        g.add_conditional_edges("tools", self._after_tools,
                                {"model": "model", "end": END})
        g.add_edge("respond", END)
        return g.compile(checkpointer=bot_checkpointer())

    # -- nodes ------------------------------------------------------------
    def _entry(self, state: BotState, config) -> dict:
        """Clear the per-turn scratch, and screen what arrived.

        Journey state is untouched. The inbound screen is the ApplyGuardrail
        call over text the model has not seen yet - a channel is an untrusted
        boundary, and WhatsApp in particular carries text from anyone who
        knows the number.
        """
        _, trace = _turn(config)
        blocked = False
        if guardrail.configured():
            text = _last_human_text(state) or ""
            if text:
                v = guardrail.screen_inbound(text)
                if v.action not in ("NOT_CONFIGURED",):
                    trace.add("guardrail", "inbound", action=v.action,
                              reasons=v.reasons, blocked=v.blocked)
                blocked = v.blocked
        return {"rounds": 0, "retrieved": [], "tool_facts": [],
                "tools_called": [], "used_core_tool": False,
                "looked_up": False, "parked": False, "last_citations": [],
                "inbound_blocked": blocked}

    def _reduce(self, state: BotState, config) -> dict:
        """Bound the thread before anything reads it.

        A checkpoint holds the whole message list; documents are kept out of
        it by `storage/documents.py`, and old turns are folded into a summary
        here. Between them a text conversation never approaches the 1 MB a
        DynamoDB query returns.
        """
        _, trace = _turn(config)
        return reducer.reduce_thread(dict(state), trace)

    def _after_entry(self, state: BotState) -> str:
        return "pending" if confirm.pending_of(dict(state)) else "model"

    def _pending(self, state: BotState, config) -> dict:
        """AG-6. The customer's answer to a call we proposed last turn."""
        ctx, trace = _turn(config)
        pending = confirm.pending_of(dict(state))
        answer = confirm.read_answer(_last_human_text(state))
        trace.add("gate", "confirmation_answer", answer=answer,
                  tool=pending["tool"], token=pending["token"])

        if answer == "no":
            return {"pending_confirmation": None,
                    "messages": [_system_msg(
                        "No problem - I have not done that. What would you "
                        "like to do instead?")]}
        if answer == "unclear":
            return {"messages": [_system_msg(pending["summary"])]}

        # The token IS the idempotency key: the same operation confirmed
        # twice returns the first result rather than writing again.
        result = self._run_tool(
            pending["tool"], pending["args"],
            ctx | {"confirmed": True, "idempotency_key": pending["token"]},
            trace)
        # A tool RESULT with no matching tool CALL is not a conversation the
        # Converse API accepts, and it is not one that happened either: the
        # model proposed this last turn and we deferred it. Replay the pair.
        return {
            "pending_confirmation": None,
            "used_core_tool": True,
            "tool_facts": [json.dumps(result, default=str)],
            "tools_called": [pending["tool"]],
            "messages": [
                AIMessage(content="", tool_calls=[{
                    "name": pending["tool"], "args": pending["args"],
                    "id": pending["token"]}]),
                ToolMessage(content=json.dumps(result, default=str)[:8000],
                            tool_call_id=pending["token"]),
            ],
        }

    def _after_pending(self, state: BotState) -> str:
        last = state["messages"][-1]
        return "model" if isinstance(last, ToolMessage) else "end"

    def _model(self, state: BotState, config) -> dict:
        cfg = settings()
        _, trace = _turn(config)
        history = model_visible(state["messages"])
        messages = ([SystemMessage(content=self._system_prompt())]
                    + history[-cfg.msg_history_to_keep:])
        with trace.timed("llm", f"invoke round {state.get('rounds', 0) + 1}",
                         model="stub" if cfg.mock_llm else cfg.bedrock_model_id,
                         tools_bound=len(self.tool_names)):
            ai: AIMessage = self.llm.invoke(messages)
        return {"messages": [ai], "rounds": state.get("rounds", 0) + 1}

    def _after_model(self, state: BotState) -> str:
        last = state["messages"][-1]
        calls = getattr(last, "tool_calls", None) or []
        if calls and state.get("rounds", 0) < MAX_TOOL_ROUNDS:
            return "tools"
        return "respond"

    def _tools(self, state: BotState, config) -> dict:
        ctx, trace = _turn(config)
        ai = state["messages"][-1]
        out: dict[str, Any] = {"messages": [], "tools_called": [],
                               "tool_facts": [], "retrieved": []}
        user_text = _last_human_text(state)

        for call in (getattr(ai, "tool_calls", None) or []):
            name, args = call["name"], (call.get("args") or {})
            spec = get_tool(name)
            out["tools_called"].append(name)

            if name == "human_handoff" and not state.get("looked_up") \
                    and not WANTS_HUMAN.search(user_text or ""):
                trace.add("gate", "handoff_before_lookup",
                          detail="refused: nothing was looked up first")
                out["messages"].append(ToolMessage(
                    content=json.dumps({
                        "error": "lookup_first",
                        "detail": ("do not hand off a question you have not "
                                   "looked up; search the approved sources "
                                   "first and hand off only if they do not "
                                   "answer it"),
                        "try_instead": self._open_tools()}),
                    tool_call_id=call.get("id", name)))
                continue

            # A mutating tool is PROPOSED, not performed. The customer is
            # asked, and the answer arrives on the next turn.
            if spec and spec.confirm and not ctx.get("confirmed"):
                parked = confirm.park(out, name, args)   # writes into `out`
                trace.add("gate", f"confirmation_required {name}",
                          token=parked["token"], args=args)
                return out | {
                    "parked": True,
                    "pending_confirmation": parked,
                    "messages": [_system_msg(parked["summary"])],
                }

            result = self._run_tool(name, args, ctx, trace)
            failed = isinstance(result, dict) and "error" in result
            if spec and spec.effect == "read" and not failed:
                out["looked_up"] = True
            if name in RETRIEVAL_TOOLS and isinstance(result, dict):
                result = self._record_retrieval(result, out, args, trace)
            elif spec and spec.authority == "core" and not failed:
                out["used_core_tool"] = True
                out["tool_facts"].append(json.dumps(result, default=str))

            out["messages"].append(ToolMessage(
                content=json.dumps(result, default=str)[:8000],
                tool_call_id=call.get("id", name)))
        return out

    def _after_tools(self, state: BotState) -> str:
        if state.get("parked"):
            return "end"
        if state.get("rounds", 0) >= MAX_TOOL_ROUNDS:
            return "end"
        return "model"

    def _respond(self, state: BotState, config) -> dict:
        ctx, trace = _turn(config)
        ai = state["messages"][-1]
        text = _text_of(ai)
        retrieved = list(state.get("retrieved") or [])
        tools_called = list(state.get("tools_called") or [])

        if state.get("rounds", 0) >= MAX_TOOL_ROUNDS and \
                (getattr(ai, "tool_calls", None) or []):
            trace.add("error", "tool_round_limit", rounds=MAX_TOOL_ROUNDS)
            self._run_tool("human_handoff",
                           {"reason": "tool round limit",
                            "summary": (_last_human_text(state) or "")[:200]},
                           ctx, trace)
            return {"messages": [_system_msg(HANDOFF)],
                    "last_tools": tools_called, "last_citations": []}

        # Facts this journey established count as established.
        known = " ".join(list(state.get("known_facts") or [])
                         + list(state.get("tool_facts") or [])).strip()
        # The guardrail runs on EVERY answer, including one produced with no
        # tool call at all - a model that skips retrieval and answers from
        # general knowledge is exactly what KB-4 exists to catch.
        # Was anything ATTEMPTED this turn? Only a tool that changes state
        # makes "I have not done that" the right thing to say; for an
        # informational turn it describes a transaction the customer never
        # started.
        action_attempted = False
        for name in tools_called:
            spec = spec_for_tool(name)
            if spec is not None and getattr(spec, "effect", "read") != "read":
                action_attempted = True
                break

        # What this conversation already confirmed WITH a source. Without it
        # a follow-up - "how many months was that again?" - is refused for
        # restating something the customer was correctly told a moment ago,
        # which is how a bot ends up unable to hold a conversation.
        established = (list(state.get("known_facts") or [])
                       + list(state.get("tool_facts") or []))

        text, report = citations.enforce(
            text, retrieved, bool(state.get("used_core_tool")),
            established=established,
            retrieval_ran=any(t in RETRIEVAL_TOOLS for t in tools_called),
            action_attempted=action_attempted)

        trace.add("guardrail", "citations",
                  cited=sorted(set(report.get("cited", []))),
                  dropped=report.get("dropped", []),
                  invented_refs=report.get("invented_refs", []),
                  verifier=report.get("verifier"),
                  refused=report.get("refused", False),
                  grounded_by=("retrieval" if retrieved
                               else "core tool" if state.get("used_core_tool")
                               else "nothing"))
        # Contextual grounding, against the passages the answer was supposed
        # to come from. This is deliberately AFTER citations.enforce: the two
        # ask different questions, and a rewritten refusal should be checked
        # in the form the customer will actually see.
        if guardrail.configured() and text and not report.get("refused"):
            passages = [str(r.get("text") or r.get("chunk") or "")
                        for r in retrieved]
            if passages:
                v = guardrail.screen_retrieved(
                    passages, _last_human_text(state) or "", text)
                if v.action not in ("NOT_CONFIGURED", "ERROR"):
                    # `alarm` rather than `blocked`, because that is what it
                    # is. A score below the threshold is worth seeing and is
                    # not worth refusing a correct answer over - the same
                    # answer scored 0.56 and 0.14 on consecutive runs.
                    trace.add("guardrail", "grounding", action=v.action,
                              grounding=v.grounding, relevance=v.relevance,
                              reasons=v.reasons, alarm=v.blocked,
                              enforced=False)
                # Grounding is RECORDED, not enforced, and that is a
                # measurement rather than a preference.
                #
                # Scored against the deployed guardrail, on one question whose
                # citations all resolved:
                #     fabricated answer, contradicting the source ... 0.00
                #     verbatim from a single passage ................ 1.00
                #     correct answer, paraphrased .................. 0.56
                #     the same correct answer, next run ............ 0.14
                #
                # Bedrock scores the WHOLE response, and this bot's answers
                # carry conversational framing - "based on the policy
                # wording", "I can look that up for you" - which appears in no
                # source document and drags the score down. 0.14 and 0.00 are
                # too close to separate a good answer from a fabricated one.
                #
                # So the RULE decides and the SCORE is watched: citations.py
                # refuses an answer whose material claims do not resolve to a
                # retrieved document, which is deterministic and does not care
                # how the sentence was phrased. Letting a probabilistic score
                # veto that would refuse correct answers, and a bot that
                # refuses correct answers is not cautious - it is broken.
                #
                # The score still reaches the trace and the audit record, so a
                # drift in grounding is visible even though it is not fatal.
                # `outputs[0].text` means two different things depending on
                # the verdict: the MASKED text when the guardrail rewrote
                # something, and the BLOCK MESSAGE when it refused. Taking it
                # unconditionally replaced a perfectly good answer with
                # "I could not give you a reliable answer" while the trace
                # still said the block was advisory - the log and the customer
                # disagreed, which is the worst kind of wrong.
                if not v.blocked and v.text and v.text != text:
                    # PII the guardrail masked. Take its version.
                    text = v.text

        trace.add("respond", "final", chars=len(text or ""), tools=tools_called)

        # Same id replaces the model's message in place, so the transcript
        # holds what the customer was actually shown.
        cleaned = AIMessage(content=text or "(no reply)", id=ai.id)
        if report.get("refused"):
            cleaned.additional_kwargs["system_authored"] = True

        facts = (list(state.get("known_facts") or [])
                 + list(state.get("tool_facts") or []))[-KNOWN_FACTS_KEPT:]
        return {"messages": [cleaned], "known_facts": facts,
                "last_tools": tools_called,
                "last_citations": sorted(set(report.get("cited", [])))}

    # -- tool execution ---------------------------------------------------
    def _open_tools(self) -> list[str]:
        """Bound tools that need no identity and change nothing - the ones a
        denied call can always be redirected to."""
        return sorted(t.name for t in list_tools(match=self.spec.tool_tags)
                      if t.auth == "anonymous" and t.effect == "read")

    def _inject(self, spec, args: dict, ctx: dict) -> dict:
        """Context the model must not supply.

        Whose data this is, which corpus it may see, and the key that makes a
        retry safe. Every one of these is an entitlement or a control, and an
        entitlement the model can pass as an argument is one it can choose.
        """
        import inspect

        params = inspect.signature(spec.fn).parameters
        injected = dict(args)
        for name, value in (("_customer_id", ctx.get("customer_id")),
                            ("_user_id", ctx.get("user_id")),
                            ("_lob", ctx.get("lob")),
                            ("_scopes", ctx.get("corpus_scope")),
                            ("_idempotency_key", ctx.get("idempotency_key"))):
            if name in params:
                injected[name] = value
        return injected

    def _run_tool(self, name: str, args: dict, ctx: dict, trace: Trace) -> Any:
        spec = get_tool(name)
        if spec is None:
            trace.add("error", "unknown_tool", name=name)
            return {"error": "unknown_tool", "name": name}

        # Capability was decided at bind time. Authority is decided here, per
        # call, because a bound tool is not permission to read anyone's data -
        # and a model can name a tool it was never offered.
        ok, why = authorize(spec, ctx, args)
        if not ok:
            trace.add("gate", f"denied {name}", reason=why,
                      persona=ctx.get("persona"), lob=ctx.get("lob"))
            # A denial that only says "no" leaves the model with one move:
            # apologise. Telling it what it MAY call turns a refusal into a
            # redirect.
            return {"error": "not_authorized", "detail": why,
                    "remedy": ("this call is not permitted for the current "
                               "caller; do not retry it with different "
                               "arguments"),
                    "try_instead": self._open_tools()}

        if spec.effect in ("write", "dispatch"):
            trace.add("gate", f"{spec.effect} {name}",
                      idempotency_key=ctx.get("idempotency_key"),
                      confirmed=ctx.get("confirmed", False),
                      idempotent=spec.idempotent)

        call_args = self._inject(spec, args, ctx)
        with trace.timed("tool", name, args=dict(args),
                         authority=spec.authority, effect=spec.effect,
                         pii=spec.pii) as t:
            try:
                result = spec.fn(**call_args)
            except TypeError as exc:
                # A bad argument set is a model error, not a server error, and
                # the model can fix it if it is told what it got wrong.
                result = {"error": "bad_arguments", "detail": str(exc),
                          "expected": json_schema(spec)}
            except Exception as exc:                         # noqa: BLE001
                result = {"error": type(exc).__name__, "detail": str(exc)}
            t.note(ok=not (isinstance(result, dict) and "error" in result))
        return result

    def _record_retrieval(self, result: dict, out: dict, args: dict,
                          trace: Trace) -> dict:
        """Pull the rejected candidates out of the tool result into the trace
        (OB-4) and hand the model only what was accepted."""
        meta = result.pop("_trace", {}) or {}
        stats = meta.get("stats", {})
        chunks = result.get("chunks", [])
        out["retrieved"].extend(chunks)
        trace.add("retrieve", args.get("query", "")[:120],
                  accepted=len(chunks), rejected=len(meta.get("rejected", [])),
                  corpus=stats.get("corpus"),
                  after_prefilter=stats.get("after_prefilter"),
                  dropped_by_prefilter=stats.get("dropped"),
                  floor=stats.get("floor"), as_of=args.get("as_of"),
                  accepted_chunks=[{k: c[k] for k in
                                    ("chunk_id", "score", "source", "section",
                                     "page", "authority", "version")}
                                   for c in chunks],
                  rejected_chunks=meta.get("rejected", []))
        return result

    # -- prompt -----------------------------------------------------------
    def _system_prompt(self) -> str:
        cfg = settings()
        return (
            f"{self.spec.system_prompt}\n\n"
            f"Retrieval scope: {', '.join(self.spec.corpus_scope)}.\n"
            f"Active line of business: {self.spec.lob}. You have no access to "
            f"any other line of business and must not speculate about one.\n"
            "When you use a retrieved passage, cite it as [chunk_id] "
            "immediately after the sentence it supports. A sentence with no "
            "citation and no tool result behind it will be removed before the "
            "customer sees it, so do not write one.\n"
            "Search the approved sources BEFORE you answer a question about "
            "cover, process or rules, and before you hand anything to a "
            "colleague. A handoff without a search is not a handoff, it is a "
            "refusal to look.\n"
            "If retrieval returns nothing, say you do not have an approved "
            "source and offer a colleague. Do not answer from general "
            "knowledge.\n"
            "Never ask for a full Aadhaar number, a card number, a CVV, a UPI "
            "PIN or a one-time password, and never repeat one back.\n"
            f"prompt_version={cfg.prompt_version} "
            f"config_version={cfg.config_version}"
        )


def _system_msg(text: str) -> AIMessage:
    """A message the SYSTEM wrote in the bot's voice.

    The customer sees it and the audit records it, but it is kept out of the
    model's own context - replaying a confirmation prompt as the model's
    prior output teaches it to write that sentence instead of calling the
    tool.
    """
    return AIMessage(content=text,
                     additional_kwargs={"system_authored": True})


def model_visible(messages) -> list:
    """History minus the messages the SYSTEM wrote in the bot's voice.

    A confirmation prompt - "I am about to issue the policy. Shall I go
    ahead?" - is generated by `confirm.py`, not by the model. Replaying it as
    an assistant message teaches the model the pattern, and it starts writing
    that sentence INSTEAD of calling the tool. The customer then says "yes",
    nothing is parked to confirm, and the journey stalls with the bot
    politely describing an action it never took.

    Those messages stay in the checkpoint and in the audit record. They are
    simply not shown back to the model as its own prior output.
    """
    return [m for m in messages
            if not getattr(m, "additional_kwargs", {}).get("system_authored")]


def _last_human_text(state: BotState) -> str | None:
    for m in reversed(state.get("messages", [])):
        if getattr(m, "type", "") == "human":
            return m.content if isinstance(m.content, str) else str(m.content)
    return None


def _text_of(ai) -> str:
    content = getattr(ai, "content", "")
    if isinstance(content, list):        # Converse returns content blocks
        return " ".join(b.get("text", "") for b in content
                        if isinstance(b, dict)).strip()
    return (content or "").strip()


_AGENTS: dict[str, Agent] = {}


def agent_for(spec: AgentSpec) -> Agent:
    if spec.key not in _AGENTS:
        _AGENTS[spec.key] = Agent(spec)
    return _AGENTS[spec.key]


def reset_agents() -> None:
    """Tests switch MOCK_LLM and checkpointers between cases."""
    _AGENTS.clear()
