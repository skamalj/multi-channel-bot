"""AG-UI events, in one place.

Two things produce this event stream and they must produce the same one:

* `agent/server.py`, inside the AgentCore Runtime container;
* `app/main.py`, when the console runs the agent in-process for local
  development with no AWS at all.

Written twice they would drift, and the drift would show up as a console that
renders correctly against a laptop and incorrectly against the deployment -
which is the worst place to discover it.

**The process streams; the claims do not.** Steps, tool calls and traces are
emitted as they happened. The ANSWER is emitted once, at the end, as a single
`TEXT_MESSAGE_CONTENT`. The citation guardrail rewrites an answer AFTER the
model finishes - an uncited premium becomes a refusal - so streaming token
deltas would put a fabricated figure on a customer's screen and retract it a
second later.
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid

log = logging.getLogger("mcb.agui")

SESSION_HEADER = "X-Amzn-Bedrock-AgentCore-Runtime-Session-Id"

# A trace event maps to a step boundary or to a note beside it. Steps are the
# ones a person recognises as "the agent is doing something".
STEP_KINDS = {"resolve", "bind", "retrieve", "gate", "respond"}


def sse(event: dict) -> str:
    return f"data: {json.dumps(event, separators=(',', ':'), default=str)}\n\n"


def ingest_event(payload: dict):
    """Turn an AG-UI RunAgentInput into the channel event the app expects."""
    from app.channels.base import IngestEvent

    messages = payload.get("messages") or []
    text = ""
    msg_id = payload.get("runId") or uuid.uuid4().hex
    for m in reversed(messages):
        if m.get("role") == "user":
            content = m.get("content")
            # CopilotKit sends content as a string; some AG-UI clients send
            # a list of parts. Both are legitimate and both arrive here.
            if isinstance(content, list):
                text = " ".join(p.get("text", "") for p in content
                                if isinstance(p, dict))
            else:
                text = content or ""
            msg_id = m.get("id") or msg_id
            break

    props = payload.get("forwardedProps") or {}
    return IngestEvent(
        message_id=msg_id,
        # threadId IS the person. The resolver thread and the bot thread are
        # derived from it, which is why a conversation that started on the
        # web continues on WhatsApp.
        user_id=payload.get("threadId") or "anonymous",
        channel=props.get("channel") or "webchat",
        channel_identity=props.get("channel_identity"),
        display_name=props.get("display_name"),
        text=text,
        entry_persona=props.get("persona"),
        entry_lob=props.get("lob"),
    )


def trace_events(trace, parent_message_id: str) -> list[dict]:
    """AG-UI events for everything a Trace recorded.

    `parent_message_id` is the assistant message the tool calls belong
    to. The protocol wants the relationship stated, not inferred.
    """
    out: list[dict] = []
    open_steps: set[str] = set()
    for ev in list(trace.events):
        detail = dict(ev.detail or {})
        if ev.ms is not None:
            detail["ms"] = ev.ms

        if ev.kind == "tool":
            # A tool call is a SEQUENCE in AG-UI, not two loose events, and
            # the schema is enforced by clients even though a raw SSE reader
            # will happily accept anything. `TOOL_CALL_RESULT` requires a
            # `messageId`; without it CopilotKit rejects the whole run with
            # "ZodError: messageId Required" and the browser shows nothing
            # while curl shows a perfectly good stream.
            call_id = f"call-{ev.seq}"
            tool_msg_id = f"toolmsg-{ev.seq}"
            out.append({"type": "TOOL_CALL_START", "toolCallId": call_id,
                        "toolCallName": ev.label,
                        "parentMessageId": parent_message_id})
            out.append({"type": "TOOL_CALL_ARGS", "toolCallId": call_id,
                        "delta": json.dumps(detail.get("args") or {},
                                            default=str)[:2000]})
            out.append({"type": "TOOL_CALL_END", "toolCallId": call_id})
            out.append({"type": "TOOL_CALL_RESULT",
                        "messageId": tool_msg_id,
                        "toolCallId": call_id,
                        "content": json.dumps(detail, default=str)[:2000]})
        elif ev.kind in STEP_KINDS:
            if ev.kind not in open_steps:
                out.append({"type": "STEP_STARTED", "stepName": ev.kind})
                open_steps.add(ev.kind)
            out.append({"type": "CUSTOM", "name": "trace",
                        "value": {"kind": ev.kind, "label": ev.label,
                                  "detail": detail}})
            out.append({"type": "STEP_FINISHED", "stepName": ev.kind})
            open_steps.discard(ev.kind)
        else:
            # guardrail, llm, channel, error - shown, not framed as steps.
            out.append({"type": "CUSTOM", "name": "trace",
                        "value": {"kind": ev.kind, "label": ev.label,
                                  "detail": detail}})
    return out


async def run(payload: dict):
    """One turn, as AG-UI events. Yields dicts, not SSE frames."""
    from app.orchestrator import handle

    t0 = time.perf_counter()
    event = ingest_event(payload)
    thread_id = payload.get("threadId") or "anonymous"
    log.info("turn thread=%s channel=%s chars=%d",
             thread_id, event.channel, len(event.text or ""))

    # `handle` is synchronous. It runs on a worker thread so the event loop
    # stays free to flush what has already been produced - without this the
    # whole stream would arrive at once when the turn finished, and a
    # streaming protocol that only delivers at the end is just a slow POST.
    holder: dict = {}

    def work():
        try:
            holder["result"] = handle(event)
        except Exception as exc:                             # noqa: BLE001
            holder["error"] = exc

    task = asyncio.get_running_loop().run_in_executor(None, work)

    yield {"type": "STEP_STARTED", "stepName": "thinking"}
    while not task.done():
        await asyncio.sleep(0.15)
    yield {"type": "STEP_FINISHED", "stepName": "thinking"}

    if "error" in holder:
        raise holder["error"]

    outbound, trace = holder["result"]
    message_id = uuid.uuid4().hex
    events = trace_events(trace, message_id)

    for ev in events:
        yield ev

    text = outbound[0].text if outbound else ""
    if not text:
        # A suppressed contact is a real outcome, not an error.
        yield {"type": "CUSTOM", "name": "trace",
               "value": {"kind": "respond", "label": "suppressed",
                         "detail": {"outbound": 0}}}
        return

    yield {"type": "TEXT_MESSAGE_START",
           "messageId": message_id, "role": "assistant"}
    yield {"type": "TEXT_MESSAGE_CONTENT",
           "messageId": message_id, "delta": text}
    yield {"type": "TEXT_MESSAGE_END", "messageId": message_id}

    yield {"type": "CUSTOM", "name": "trace",
           "value": {"kind": "respond", "label": "done",
                     "detail": {"ms": round((time.perf_counter() - t0) * 1000, 1),
                                "trace_id": trace.trace_id,
                                "events": len(trace.events)}}}


async def stream(payload: dict):
    """The full SSE body for one turn, framed and error-wrapped."""
    thread_id = payload.get("threadId") or "anonymous"
    run_id = payload.get("runId") or uuid.uuid4().hex
    yield sse({"type": "RUN_STARTED", "threadId": thread_id, "runId": run_id})
    try:
        async for ev in run(payload):
            yield sse(ev)
        yield sse({"type": "RUN_FINISHED",
                   "threadId": thread_id, "runId": run_id})
    except Exception as exc:                                 # noqa: BLE001
        log.exception("turn failed")
        # AGENT_ERROR is the runtime category: the stream has already
        # started, so this still returns HTTP 200.
        yield sse({"type": "RUN_ERROR", "code": "AGENT_ERROR",
                   "message": f"{type(exc).__name__}: {exc}"})
