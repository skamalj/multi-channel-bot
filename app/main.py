"""FastAPI app: the web-chat test harness plus the glass-box trace.

WhatsApp is bolted on later by registering WhatsAppAdapter and pointing the
Meta webhook at POST /webhook/whatsapp - nothing below the channel layer
changes. The parsing is real; the outbound send is not wired, and the
webhook below says so rather than pretending.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import (FileResponse, JSONResponse,
                               StreamingResponse)
from pydantic import BaseModel

from app.agents.registry import REGISTRY, capability_matrix
from app.channels.webchat import WebChatAdapter
from app.channels.whatsapp import WhatsAppAdapter
from app.config import settings
from app.knowledge.corpus import stats as corpus_stats
from app.mcpserver.server import describe
from app.memory.longterm import consent, interactions
from app.memory.registry import declared as declared_stores, erase
from app.obs import audit
from app.orchestrator import handle
from app.resolver.graph import hydrate, resolver_graph, thread_for
from app.resolver.store import ResolverSession

logging.basicConfig(level=settings().log_level)
log = logging.getLogger("mcb")

app = FastAPI(title="Protec multi-channel bot", version="0.2.0")
WEB = Path(__file__).resolve().parents[1] / "web"

webchat = WebChatAdapter()
whatsapp = WhatsAppAdapter()


class ChatIn(BaseModel):
    user_id: str
    text: str
    persona: str | None = None      # entry context, as a deep link would carry
    lob: str | None = None
    display_name: str | None = None


class ConsentIn(BaseModel):
    user_id: str
    purpose: str
    granted: bool = True


class StepUpIn(BaseModel):
    user_id: str


def _session(user_id: str) -> ResolverSession:
    """Read the resolver thread. The checkpointer is the store."""
    cfg = {"configurable": {"thread_id": thread_for(user_id)}}
    return hydrate(resolver_graph().get_state(cfg).values.get("session"),
                   user_id)


def _save_session(user_id: str, session: ResolverSession) -> None:
    resolver_graph().update_state(
        {"configurable": {"thread_id": thread_for(user_id)}},
        {"session": session.model_dump()})


@app.get("/")
def index():
    return FileResponse(WEB / "index.html")


@app.get("/api/health")
def health():
    cfg = settings()
    return {"ok": True, "region": cfg.aws_region, "mock_llm": cfg.mock_llm,
            "model": "stub" if cfg.mock_llm else cfg.bedrock_model_id,
            "small_model": "stub" if cfg.mock_llm else cfg.bedrock_small_model_id,
            "prompt_version": cfg.prompt_version,
            "config_version": cfg.config_version,
            "corpus": corpus_stats()}


@app.get("/api/bots")
def bots():
    return {"bots": [s.model_dump() for s in REGISTRY.values()],
            "capability_matrix": capability_matrix()}


@app.get("/api/tools")
def tools():
    return {"tools": describe()}


@app.get("/api/memory")
def memory():
    """ME-8: every store, its key, TTL, sensitivity and erasure handler."""
    return {"stores": declared_stores()}


@app.get("/api/audit/{user_id}")
def audit_for(user_id: str, limit: int = 50):
    """OB-3: one record per turn, keyed by person, spanning every LOB."""
    return {"user_id": user_id, "entries": audit.for_person(user_id, limit)}


@app.get("/api/session/{user_id}")
def session_for(user_id: str):
    """The resolver session: route ledger, shared profile, auth state."""
    s = _session(user_id)
    return {"session": s.model_dump(),
            "interactions": interactions.get(user_id),
            "consent": consent.current(user_id)}


@app.post("/api/consent")
def set_consent(body: ConsentIn):
    """Grant or withdraw a purpose. Append-only: a withdrawal never deletes
    the grant that preceded it."""
    consent.record(body.user_id, body.purpose, body.granted,
                   {"source": "user_stated", "ref": "web console"})
    return {"user_id": body.user_id, "consent": consent.current(body.user_id)}


@app.post("/api/stepup")
def step_up(body: StepUpIn):
    """RS-9: stand-in for an OTP or a re-authentication.

    The real one is a second factor. What matters here is that it is a
    SEPARATE act from the conversation - a persona change cannot be talked
    into existence."""
    s = _session(body.user_id)
    s.profile.authenticated = True
    s.persona_stepup_at = time.time()
    _save_session(body.user_id, s)
    return {"user_id": body.user_id, "authenticated": True,
            "stepped_up_at": s.persona_stepup_at}


@app.post("/api/erase/{user_id}")
def erase_user(user_id: str):
    """ME-8: run every registered erasure handler and report per store.

    The audit store reports `retained` rather than a number - an erasure
    request does not remove a regulatory record, and saying so is better than
    a zero that quietly means nothing was there."""
    return {"user_id": user_id, "report": erase(user_id)}


@app.post("/api/chat")
def chat(body: ChatIn):
    t0 = time.perf_counter()
    events = webchat.parse(body.model_dump())
    replies, trace = handle(events[0])
    return JSONResponse({
        "replies": [r.model_dump() for r in replies],
        "trace": trace.model_dump(),
        "turn_ms": round((time.perf_counter() - t0) * 1000, 1),
    })


# --- WhatsApp: parsing is real, sending is not yet wired -------------------
@app.get("/webhook/whatsapp")
def wa_verify(request: Request):
    q = request.query_params
    # Echo the challenge only when the verify token matches. The token check
    # and the raw-body signature check (CH-6) land with the outbound wiring.
    return JSONResponse(content=int(q.get("hub.challenge", 0)))


@app.post("/webhook/whatsapp")
async def wa_inbound(request: Request):
    # TODO when bolting on: verify X-Hub-Signature-256 over the RAW body,
    # dedupe on message id, enqueue, and ack within 300 ms (CH-5, CH-6).
    raw = await request.json()
    events = whatsapp.parse(raw)
    for ev in events:
        handle(ev)
    return {"status": "ok", "parsed": len(events),
            "note": "inbound parsing is live; outbound send is not wired"}


# ---------------------------------------------------------------------------
# AG-UI
# ---------------------------------------------------------------------------
# The console is an AG-UI CLIENT and this is what it talks to. AG-UI is a
# protocol with two sides: the agent EMITS the events and something with a
# screen CONSUMES them. The emitting half lives in `app/agui.py` and in the
# Runtime container; this is the seam that lets a browser reach either.
#
# Two modes, and the difference is only where the turn runs:
#
#   remote - `agent_runtime_arn` is set. The request is signed with SigV4 and
#            proxied to the deployed AgentCore Runtime, and the response is
#            streamed through UNCHANGED. This exists because a browser cannot
#            hold AWS credentials, so something has to sign, and that is the
#            whole reason "FastAPI for both" was the design.
#
#   local  - no ARN. The agent runs in this process. Same events, no AWS.
#
# The console cannot tell which it is talking to, which is the point: what it
# renders against a laptop is what it renders against the deployment.
@app.post("/agui")
async def agui(request: Request) -> StreamingResponse:
    from app.agui import stream

    payload = await request.json()
    cfg = settings()

    # WHO is asking, decided here rather than taken from the payload.
    #
    # CopilotKit's `threadId` prop controls ITS transcript, not the AG-UI
    # RunAgentInput it sends onward - the agent kept receiving a fresh UUID
    # per run, so consecutive turns landed on different resolver and bot
    # threads and the agent could not remember the previous sentence. The
    # whole two-checkpointer design rests on the thread being the person.
    #
    # So the identity travels in a header the console sets and this reads,
    # end to end, and overrides whatever the client put in `threadId`.
    user = request.headers.get("x-mcb-user")
    if user:
        payload["threadId"] = user

    if not cfg.agent_runtime_arn:
        return StreamingResponse(
            stream(payload), media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    return StreamingResponse(
        _proxy_to_runtime(payload, cfg), media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


def _proxy_to_runtime(payload: dict, cfg):
    """Sign, invoke, and pass the event stream straight through.

    Nothing is parsed or re-emitted here. A proxy that rebuilt the events
    would be a second implementation of the protocol, and the console would
    be testing the proxy rather than the agent.
    """
    import hashlib
    import json as _json

    import boto3

    thread = payload.get("threadId") or "anonymous"
    # runtimeSessionId has a 33-character minimum and our user ids are phone
    # numbers, so it is derived rather than padded - and derived from the
    # thread, so the same person keeps the same runtime session.
    session = hashlib.sha256(f"console:{thread}".encode()).hexdigest()

    try:
        resp = boto3.client("bedrock-agentcore",
                            region_name=cfg.aws_region).invoke_agent_runtime(
            agentRuntimeArn=cfg.agent_runtime_arn,
            runtimeSessionId=session,
            qualifier=cfg.agent_qualifier,
            payload=_json.dumps(payload).encode())
    except Exception as exc:                                 # noqa: BLE001
        log.exception("runtime invocation failed")
        yield ("data: " + _json.dumps({
            "type": "RUN_ERROR", "code": "AGENT_ERROR",
            "message": f"{type(exc).__name__}: {exc}"}) + "\n\n")
        return

    body = resp.get("response")
    for chunk in body:
        if not chunk:
            continue
        yield chunk.decode() if isinstance(chunk, (bytes, bytearray)) else chunk


# The last turn's trace, per thread.
#
# The glass box needs these and CopilotKit's client does not forward AG-UI
# CUSTOM events to a subscriber - the chat renders, the trace never arrives.
# Rather than keep guessing at another library's internals, the bridge keeps
# what it already produced and the console asks for it.
#
# Honest about what this is: the trace is FETCHED after the turn rather than
# streamed during it. The events and their order are the real ones; the
# liveness is not. Bounded per thread so a long conversation cannot grow it
# without limit.
_LAST_TRACE: dict[str, list] = {}
_TRACE_THREADS = 32


def _remember_trace(thread: str, events: list) -> None:
    if len(_LAST_TRACE) >= _TRACE_THREADS and thread not in _LAST_TRACE:
        _LAST_TRACE.pop(next(iter(_LAST_TRACE)), None)
    _LAST_TRACE[thread] = events
    # Kept as a fallback for any client that has no stable thread of its own.
    # The console no longer needs it - CopilotKitProvider takes a threadId
    # prop, so the identity chosen in the UI IS the thread.
    _LAST_TRACE["__last"] = events


@app.get("/api/agui/trace/{thread}")
def agui_trace(thread: str) -> dict:
    return {"thread": thread, "events": _LAST_TRACE.get(thread, [])}


@app.get("/api/agui/mode")
def agui_mode() -> dict:
    """Which agent the console is actually talking to.

    Worth surfacing: a console that silently ran the agent in-process would
    look identical to one driving the deployment, and would prove nothing.
    """
    cfg = settings()
    return {"mode": "remote" if cfg.agent_runtime_arn else "local",
            "runtime_arn": cfg.agent_runtime_arn or None,
            "qualifier": cfg.agent_qualifier}
