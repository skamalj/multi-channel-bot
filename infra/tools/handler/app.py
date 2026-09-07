"""The tool server, as one Lambda behind an AgentCore Gateway.

The Gateway does not pass the tool name in the event. It passes it in the
client context:

    context.client_context.custom['bedrockAgentCoreToolName']

and it prefixes the target name with `___`, so `mcbcore___policy_get` means
the tool `policy_get`. Splitting on that delimiter is the whole dispatch.

**Entitlements are not arguments.** `_customer_id`, `_user_id`, `_lob`,
`_scopes` and `_idempotency_key` come from the verified JWT claims that the
Gateway forwards - never from the model. A tool that took its own scope as a
parameter would let the model widen its own access by asking nicely, which is
the failure this design exists to prevent.
"""
from __future__ import annotations

import json
import logging
import os

log = logging.getLogger()
log.setLevel(logging.INFO)

DELIMITER = "___"


def _tool_name(context) -> str | None:               # noqa: ANN001
    """The tool the Gateway is asking for, or None if not invoked by one."""
    client = getattr(context, "client_context", None)
    custom = getattr(client, "custom", None) or {}
    raw = custom.get("bedrockAgentCoreToolName")
    if not raw:
        return None
    return raw.split(DELIMITER)[-1]


def _claims(event: dict) -> dict:
    """Caller identity, from the token the Gateway validated.

    The Gateway has already checked the signature, the issuer, the audience
    and the client id before anything reaches here, so these claims are
    trustworthy in a way that nothing in `event` from the model ever is.
    """
    ctx = (event or {}).get("requestContext") or {}
    authorizer = ctx.get("authorizer") or {}
    return authorizer.get("claims") or authorizer.get("jwt", {}).get("claims") or {}


def _entitlements(event: dict) -> dict:
    claims = _claims(event)
    return {
        "_user_id": claims.get("sub") or claims.get("username"),
        "_customer_id": claims.get("custom:customer_id"),
        "_lob": claims.get("custom:lob"),
        "_scopes": (claims.get("scope") or "").split(),
        "_client_id": claims.get("client_id"),
    }


def handler(event, context):                          # noqa: ANN001
    """Dispatch one tool call.

    Until the core store is ported to Redshift this reports what it WOULD do
    rather than pretending to have done it. A stub that invents a policy
    number is worse than one that says it is a stub - the citation guardrail
    downstream is built on the assumption that a tool result is a fact.
    """
    tool = _tool_name(context)
    ent = _entitlements(event or {})
    log.info("tool=%s user=%s scopes=%s", tool, ent.get("_user_id"),
             ent.get("_scopes"))

    if tool is None:
        # Not a Gateway invocation. Used by the deploy pipeline to prove the
        # function is reachable and correctly configured before any tool
        # exists to call.
        return {
            "ok": True,
            "probe": True,
            "redshift_configured": bool(os.environ.get("REDSHIFT_HOST")),
            "documents_bucket": os.environ.get("DOCUMENTS_BUCKET"),
        }

    return {
        "ok": False,
        "tool": tool,
        "error": "not_yet_ported",
        "detail": (
            f"{tool} is registered with the Gateway but its implementation "
            "has not been moved onto Redshift yet."),
        "entitlements_seen": {k: v for k, v in ent.items() if v},
        "arguments": {k: v for k, v in (event or {}).items()
                      if not k.startswith("_") and k != "requestContext"},
    }


if __name__ == "__main__":                            # pragma: no cover
    print(json.dumps(handler({}, None), indent=2))
