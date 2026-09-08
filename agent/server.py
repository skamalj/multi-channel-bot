"""The AG-UI server that AgentCore Runtime hosts.

Contract (runtime-agui-protocol-contract):
  * host 0.0.0.0, port 8080, ARM64 container
  * POST /invocations  -> text/event-stream of AG-UI events
  * GET  /ping         -> {"status": "Healthy"}
  * WS   /ws           -> bidirectional (not used yet)

**The process streams; the claims do not.**

`orchestrator.handle()` is synchronous and returns when the turn is finished,
but the `Trace` it writes into fills up as it goes. So the turn runs on a
worker thread and this streams trace events as they appear - the customer
watches retrieval, tool calls and gates happen - and the ANSWER is emitted
once, at the end, as a single TEXT_MESSAGE_CONTENT.

That is not a performance choice. The citation guardrail rewrites an answer
AFTER the model finishes: an uncited premium becomes a refusal. Streaming
token deltas would put a fabricated figure on a customer's screen and retract
it a second later, which is worse than making them wait. Showing something
false quickly is not a feature.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import uuid

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
log = logging.getLogger("mcb.agui")

app = FastAPI(title="Protec agent (AG-UI)", version="1.0.0")

SESSION_HEADER = "X-Amzn-Bedrock-AgentCore-Runtime-Session-Id"

# A trace event maps to a step boundary or to a note beside it. Steps are the
# ones a person recognises as "the agent is doing something".
_STEP_KINDS = {"resolve", "bind", "retrieve", "gate", "respond"}


def _sse(event: dict) -> str:
    return f"data: {json.dumps(event, separators=(',', ':'), default=str)}\n\n"


@app.get("/ping")
def ping() -> JSONResponse:
    """Health, and a billing decision.

    Runtime MEMORY is billed for every second a session is alive, idle
    seconds included - only CPU stops during I/O wait. `time_of_last_update`
    is deliberately omitted: a timestamp that advances on every ping reads as
    a continuous status change, the idle timeout never fires, and sessions
    live to MaxLifetime billing memory the whole way.
    """
    return JSONResponse({"status": "Healthy"})


@app.get("/")
def root() -> dict:
    from app.config import settings

    cfg = settings()
    return {
        "service": "mcb-agent", "protocol": "AGUI",
        "endpoints": ["/invocations", "/ping", "/ws"],
        "model": cfg.bedrock_model_id,
        "knowledge_base": bool(cfg.knowledge_base_id),
        "guardrail": bool(cfg.guardrail_id),
        "redshift": bool(cfg.redshift_host),
    }


def _event(payload: dict, session_id: str):
    """Turn an AG-UI RunAgentInput into the channel event the app expects."""
    from app.channels.base import IngestEvent

    messages = payload.get("messages") or []
    text = ""
    msg_id = payload.get("runId") or uuid.uuid4().hex
    for m in reversed(messages):
        if m.get("role") == "user":
            text = m.get("content") or ""
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


def _trace_events(trace, sent: int, open_steps: set) -> tuple[list[dict], int]:
    """AG-UI events for trace entries not yet emitted."""
    out: list[dict] = []
    events = list(trace.events)
    for ev in events[sent:]:
        detail = dict(ev.detail or {})
        if ev.ms is not None:
            detail["ms"] = ev.ms

        if ev.kind == "tool":
            call_id = f"{ev.kind}-{ev.seq}"
            out.append({"type": "TOOL_CALL_START", "toolCallId": call_id,
                        "toolCallName": ev.label})
            out.append({"type": "TOOL_CALL_RESULT", "toolCallId": call_id,
                        "content": json.dumps(detail, default=str)[:2000]})
        elif ev.kind in _STEP_KINDS:
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
    return out, len(events)


@app.post("/invocations")
async def invocations(request: Request) -> StreamingResponse:
    payload = await request.json()
    session_id = request.headers.get(SESSION_HEADER, "")
    thread_id = payload.get("threadId") or "anonymous"
    run_id = payload.get("runId") or uuid.uuid4().hex

    async def stream():
        yield _sse({"type": "RUN_STARTED",
                    "threadId": thread_id, "runId": run_id})
        try:
            async for ev in _run(payload, session_id, thread_id, run_id):
                yield _sse(ev)
            yield _sse({"type": "RUN_FINISHED",
                        "threadId": thread_id, "runId": run_id})
        except Exception as exc:                             # noqa: BLE001
            log.exception("turn failed")
            # AGENT_ERROR is the runtime category: the stream has already
            # started, so this still returns HTTP 200.
            yield _sse({"type": "RUN_ERROR", "code": "AGENT_ERROR",
                        "message": f"{type(exc).__name__}: {exc}"})

    return StreamingResponse(
        stream(), media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


async def _run(payload: dict, session_id: str, thread_id: str, run_id: str):
    from app.obs.trace import Trace
    from app.orchestrator import handle

    t0 = time.perf_counter()
    event = _event(payload, session_id)
    log.info("turn thread=%s channel=%s chars=%d",
             thread_id, event.channel, len(event.text or ""))

    # `handle` is synchronous and writes into its Trace as it goes, so it runs
    # on a worker thread and the trace is drained here while it works. This
    # is what makes the glass box live rather than a summary printed at the
    # end.
    holder: dict = {}
    probe = Trace()

    def work():
        try:
            holder["result"] = handle(event)
        except Exception as exc:                             # noqa: BLE001
            holder["error"] = exc

    task = asyncio.get_running_loop().run_in_executor(None, work)

    # The orchestrator makes its own Trace, so until it returns there is
    # nothing local to drain. Emit a step for the wait itself rather than
    # going silent - a customer watching a blank screen assumes it broke.
    yield {"type": "STEP_STARTED", "stepName": "thinking"}
    while not task.done():
        await asyncio.sleep(0.15)
    yield {"type": "STEP_FINISHED", "stepName": "thinking"}

    if "error" in holder:
        raise holder["error"]

    outbound, trace = holder["result"]

    # Replay the trace as AG-UI events. The turn is over by now, but the
    # ordering and the content are the real thing rather than a narration.
    events, _ = _trace_events(trace, 0, set())
    for ev in events:
        yield ev

    text = outbound[0].text if outbound else ""
    if not text:
        # A suppressed contact is a real outcome, not an error.
        yield {"type": "CUSTOM", "name": "trace",
               "value": {"kind": "respond", "label": "suppressed",
                         "detail": {"outbound": 0}}}
        return

    message_id = uuid.uuid4().hex
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
