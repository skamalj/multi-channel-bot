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
from fastapi.responses import FileResponse, JSONResponse
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
