"""Tool authorization, as ONE hook.

Binding decided CAPABILITY - may this bot ever hold this tool - at build time,
by tags (app/mcpserver/registry.py). This decides AUTHORITY - may THIS caller
make THIS call, now - on every tool call, from the tool's own `extras` policy
and the per-call `runtime.context`. A refusal is returned as the tool's
result, so the framework threads it back paired to the call and the model
learns it was not allowed the way it learns a quote is not ratable: in its own
words, on the next turn. Nothing here is in the model's schema.

The interception point (`wrap_tool_call`), the short-circuit (return a
ToolMessage instead of calling the handler) and the pairing are all the
framework's. Only the decision below is ours, because only we know that
`commission_statement` may be read for your own producer id and no one
else's.
"""
from __future__ import annotations

from typing import Callable

from langchain.agents.middleware import AgentMiddleware
from langchain.messages import ToolMessage
from langchain.tools.tool_node import ToolCallRequest
from langgraph.types import Command
import json

from app.agents.context import RequestContext
from app.mcpserver.registry import get_tool


class ToolControls(AgentMiddleware):
    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command],
    ) -> ToolMessage | Command:
        extras = getattr(request.tool, "extras", None) or {}
        ctx: RequestContext = request.runtime.context
        args = request.tool_call.get("args") or {}

        name = request.tool_call["name"]
        trace = getattr(ctx, "trace", None)

        # Handoff is the FALLBACK, not the first move. Some models reach for it
        # on any question that sounds regulated, handing a colleague something
        # the corpus answers in one call. The prompt asks for a lookup first;
        # this enforces it, because a control that depends on the model reading
        # an instruction is not a control. Once anything has been looked up on
        # this thread, a handoff is free.
        if name == "human_handoff" and not _looked_up(request.state):
            if trace is not None:
                trace.add("gate", "handoff_before_lookup",
                          detail="refused: nothing was looked up first")
            return ToolMessage(
                content=json.dumps({
                    "error": "lookup_first",
                    "detail": ("do not hand off a question you have not looked "
                               "up; search the approved sources first and hand "
                               "off only if they do not answer it"),
                    "try_instead": _open_tools(request.runtime)}),
                tool_call_id=request.tool_call["id"], name=name, status="error")

        ok, why = _authorize(extras, ctx, args)
        if not ok:
            if trace is not None:
                trace.add("gate", f"denied {name}", reason=why,
                          persona=getattr(ctx, "persona", None),
                          lob=getattr(ctx, "lob", None))
            return ToolMessage(
                content=json.dumps(_refusal(extras, why, request.runtime)),
                tool_call_id=request.tool_call["id"],
                name=name,
                status="error",
            )

        # The per-tool glass box lives here now - the one place every tool call
        # passes through. A write is labelled "<effect> <name>" so the trace
        # reads the way it did when a graph node ran the tool.
        effect = extras.get("effect", "read")
        result = handler(request)
        if trace is not None:
            label = f"{effect} {name}" if effect in ("write", "dispatch") else name
            trace.add("tool", label, effect=effect,
                      authority=extras.get("authority", "none"),
                      pii=extras.get("pii", False),
                      ok=getattr(result, "status", None) != "error")
        return result


# -- the handoff gate --------------------------------------------------------

def _looked_up(state) -> bool:
    """Has a read tool run successfully anywhere on this thread? A handoff
    before any lookup is a refusal to look, not caution."""
    import json as _json

    for m in (state.get("messages") if isinstance(state, dict) else None) or []:
        if type(m).__name__ != "ToolMessage":
            continue
        tool = get_tool(getattr(m, "name", "") or "")
        if not tool or (tool.extras or {}).get("effect", "read") != "read":
            continue
        try:
            payload = _json.loads(m.content) if isinstance(m.content, str) else {}
        except Exception:                                      # noqa: BLE001
            payload = {}
        if not (isinstance(payload, dict) and "error" in payload):
            return True
    return False


# -- the decision ------------------------------------------------------------

def _authorize(extras: dict, ctx: RequestContext, args: dict) -> tuple[bool, str]:
    """Runs on every call regardless of what was bound, and takes the ARGUMENTS
    as well as the context - "may you call commission_statement" and "may you
    call it for producer P-9999" are different questions, and only the second
    is the breach."""
    if extras.get("auth") == "authenticated" and not ctx.authenticated:
        return False, "authentication required"
    if extras.get("auth") == "identified" and not ctx.user_id:
        return False, "identity required"

    purpose = extras.get("consent_purpose")
    if purpose and not ctx.consent.get(purpose):
        return False, f"consent for '{purpose}' not current"

    tags = extras.get("tags") or {}
    persona_tag = tags.get("persona")
    if persona_tag not in ("*", None):
        if ctx.persona not in {v.strip() for v in persona_tag.split("|")}:
            return False, "persona not permitted for this tool"

    # The lob tag is a build-time filter, but a model can name a tool from the
    # other compartment. Re-check it against the ACTIVE lob.
    lob_tag = tags.get("lob")
    if lob_tag not in ("*", None) and ctx.lob:
        if ctx.lob not in {v.strip() for v in lob_tag.split("|")}:
            return False, "tool is out of scope for the active line of business"

    return _subject_check(extras.get("subject", "none"), ctx, args)


def _subject_check(subject: str, ctx: RequestContext, args: dict) -> tuple[bool, str]:
    """Having the tool bound is not permission to read someone else's book,
    policy or application."""
    from app.coremock import store as core

    if subject == "producer_id":
        asked = str(args.get("producer_id", "")).strip()
        mine = str(ctx.producer_id or "").strip()
        if not mine:
            return False, "caller is not a registered producer"
        if asked and asked != mine:
            return False, ("producer_id does not match the authenticated "
                           "producer")
        return True, ""

    if subject == "policy_id":
        pol = core.get_policy(str(args.get("policy_id", "")), lob=ctx.lob or "")
        if pol.get("error") in ("not_found", "out_of_scope"):
            return True, ""            # the tool itself returns the typed error
        if ctx.persona == "agent":
            return True, ""            # a producer services their own book
        if pol.get("customer_id") != ctx.customer_id:
            return False, "policy does not belong to the authenticated customer"
        return True, ""

    if subject == "application_id":
        if not ctx.user_id:
            return False, "identity required"
        app = core._APPLICATIONS.get(str(args.get("application_id", "")))
        if not app or ctx.persona == "agent":
            return True, ""            # tool returns the typed not_found
        if app.get("user_id") and app["user_id"] != ctx.user_id:
            return False, ("application does not belong to the authenticated "
                           "caller")
        return True, ""

    return True, ""


# -- how a refusal reads to the model ----------------------------------------

def _refusal(extras: dict, why: str, runtime) -> dict:
    """A denial that only says "no" leaves the model one move: apologise.
    Telling it what it MAY do turns a refusal into a redirect. Consent is the
    one denial the customer can lift themselves, so it reads differently -
    needing permission is not the same as not being allowed."""
    if extras.get("consent_purpose") and "consent" in why:
        return {
            "error": "consent_required",
            "purpose": extras["consent_purpose"],
            "detail": why,
            "remedy": ("the customer has not agreed to this yet. Ask them, call "
                       "consent_grant with this purpose, then make this call "
                       "again. This is a permission they can give, not a fault."),
            "try_instead": ["consent_grant"],
        }
    return {
        "error": "not_authorized",
        "detail": why,
        "remedy": "not permitted for this caller; do not retry with different args",
        "try_instead": _open_tools(runtime),
    }


def _open_tools(runtime) -> list[str]:
    """Tools that need no identity and change nothing - always a safe redirect.
    Read straight off the runtime's bound tool set."""
    out = []
    for t in getattr(runtime, "tools", None) or []:
        ex = getattr(t, "extras", None) or {}
        if ex.get("auth", "anonymous") == "anonymous" and ex.get("effect", "read") == "read":
            out.append(t.name)
    return sorted(out)
