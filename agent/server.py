"""The AG-UI server that AgentCore Runtime hosts.

Contract (runtime-agui-protocol-contract):
  * host 0.0.0.0, port 8080, ARM64 container
  * POST /invocations  -> text/event-stream of AG-UI events
  * GET  /ping         -> {"status": "Healthy"}

The events themselves come from `app.agui`, which the console also uses when
it runs the agent in-process. One implementation, so a console that renders
correctly against a laptop renders correctly against the deployment.
"""
from __future__ import annotations

import logging
import os

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from app.agui import SESSION_HEADER, stream

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
log = logging.getLogger("mcb.agui.server")

app = FastAPI(title="Protec agent (AG-UI)", version="1.1.0")


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
    return {"service": "mcb-agent", "protocol": "AGUI",
            "endpoints": ["/invocations", "/ping"],
            "model": cfg.bedrock_model_id,
            "knowledge_base": bool(cfg.knowledge_base_id),
            "guardrail": bool(cfg.guardrail_id),
            "redshift": bool(cfg.redshift_host)}


@app.post("/invocations")
async def invocations(request: Request) -> StreamingResponse:
    payload = await request.json()
    session = request.headers.get(SESSION_HEADER, "")
    log.info("invocation session=%s", session[:12])
    return StreamingResponse(
        stream(payload), media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
