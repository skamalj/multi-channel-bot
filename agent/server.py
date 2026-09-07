"""The AG-UI server that AgentCore Runtime hosts.

Contract (runtime-agui-protocol-contract):
  * host 0.0.0.0, port 8080, ARM64 container
  * POST /invocations  -> text/event-stream of AG-UI events
  * GET  /ping         -> {"status": "Healthy"}
  * WS   /ws           -> bidirectional (not used yet)

**The process streams; the claims do not.**

Steps, retrieval and tool calls are emitted live, because watching the agent
work is the whole point of the glass box. The ANSWER is emitted once, as a
single TEXT_MESSAGE_CONTENT, only after the citation guardrail has passed it.

That is not a performance choice. The guardrail rewrites an answer after the
model finishes - an uncited premium becomes a refusal - so streaming deltas
would put a fabricated figure on a customer's screen and retract it a second
later. Showing something false quickly is worse than showing something true
slightly later.
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

app = FastAPI(title="Protec agent (AG-UI)", version="0.3.0")

SESSION_HEADER = "X-Amzn-Bedrock-AgentCore-Runtime-Session-Id"


def _sse(event: dict) -> str:
    """One AG-UI event, SSE-framed."""
    return f"data: {json.dumps(event, separators=(',', ':'))}\n\n"


@app.get("/ping")
def ping() -> JSONResponse:
    """Health, and a billing decision.

    Runtime MEMORY is billed for every second a session is alive, idle
    seconds included - only CPU stops during I/O wait. `time_of_last_update`
    is deliberately omitted: a timestamp that advances on every ping reads as
    a continuous status change, the idle timeout never fires, and sessions
    live to MaxLifetime (8h) billing memory the whole way.
    """
    return JSONResponse({"status": "Healthy"})


@app.post("/invocations")
async def invocations(request: Request) -> StreamingResponse:
    """Run one turn and stream AG-UI events."""
    payload = await request.json()
    session_id = request.headers.get(SESSION_HEADER, "")

    thread_id = payload.get("threadId") or "unknown"
    run_id = payload.get("runId") or uuid.uuid4().hex
    messages = payload.get("messages") or []
    text = ""
    for m in reversed(messages):
        if m.get("role") == "user":
            text = m.get("content") or ""
            break

    log.info("run thread=%s run=%s session=%s chars=%d",
             thread_id, run_id, session_id[:12], len(text))

    async def stream():
        yield _sse({"type": "RUN_STARTED",
                    "threadId": thread_id, "runId": run_id})
        try:
            async for ev in _turn(thread_id, run_id, text, payload):
                yield _sse(ev)
            yield _sse({"type": "RUN_FINISHED",
                        "threadId": thread_id, "runId": run_id})
        except Exception as exc:                             # noqa: BLE001
            log.exception("turn failed")
            # AGENT_ERROR is the only runtime-category error: the stream has
            # already started, so this still returns HTTP 200.
            yield _sse({"type": "RUN_ERROR", "code": "AGENT_ERROR",
                        "message": f"{type(exc).__name__}: {exc}"})

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


async def _turn(thread_id: str, run_id: str, text: str, payload: dict):
    """One turn's events.

    The orchestrator is not wired in yet. What IS real here is the event
    shape, the ordering, and the rule that the answer is emitted once at the
    end - so when `orchestrator.handle()` replaces the middle, the contract
    the browser depends on does not change.
    """
    t0 = time.perf_counter()

    # --- the process: streamed live ------------------------------------
    yield {"type": "STEP_STARTED", "stepName": "resolve"}
    await asyncio.sleep(0)
    yield {"type": "CUSTOM", "name": "trace",
           "value": {"kind": "resolve", "label": "route",
                     "detail": {"thread": thread_id,
                                "chars": len(text)}}}
    yield {"type": "STEP_FINISHED", "stepName": "resolve"}

    yield {"type": "STEP_STARTED", "stepName": "bind"}
    yield {"type": "CUSTOM", "name": "trace",
           "value": {"kind": "bind", "label": "configuration",
                     "detail": {"stub": True,
                                "gateway": bool(os.getenv("GATEWAY_URL")),
                                "kb": bool(os.getenv("KNOWLEDGE_BASE_ID"))}}}
    yield {"type": "STEP_FINISHED", "stepName": "bind"}

    # --- the answer: one message, after the guardrail would have run ----
    answer = (
        "The agent container is deployed and speaking AG-UI, but the "
        "orchestrator is not wired into it yet, so there is no real answer "
        "to give you. Nothing here has consulted a product document or a "
        "policy record."
    )
    message_id = uuid.uuid4().hex
    yield {"type": "TEXT_MESSAGE_START",
           "messageId": message_id, "role": "assistant"}
    yield {"type": "TEXT_MESSAGE_CONTENT",
           "messageId": message_id, "delta": answer}
    yield {"type": "TEXT_MESSAGE_END", "messageId": message_id}

    yield {"type": "CUSTOM", "name": "trace",
           "value": {"kind": "respond", "label": "done",
                     "detail": {"ms": round((time.perf_counter() - t0) * 1000, 1),
                                "citations": []}}}


@app.get("/")
def root() -> dict:
    return {"service": "mcb-agent", "protocol": "AGUI",
            "endpoints": ["/invocations", "/ping", "/ws"]}
