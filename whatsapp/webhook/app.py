"""Meta's WhatsApp webhook. Verifies, parses, enqueues - and nothing else.

This is the only publicly reachable thing in the whole build, so it does the
least possible: it proves the request came from Meta, turns a message into one
SQS message, and returns 200. No model, no database, no agent. A slow or
failing webhook makes Meta retry and eventually disable the subscription, so
the work belongs behind the queue rather than in front of it.

**The signature check is not optional.** A Lambda Function URL with
`AuthType: NONE` accepts a request from anyone who finds the address. Without
verifying `X-Hub-Signature-256` this endpoint would let a stranger inject
messages that the agent would answer as though they came from a customer -
and the agent trusts the phone number it is handed to decide who it is
talking to.

Two paths, both required by Meta:

  GET  - subscription handshake, echo `hub.challenge` if the verify token
         matches.
  POST - message delivery. Always 200, even for shapes we ignore: a non-200
         is a retry, and a retry of something we cannot parse is a loop.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os

import boto3

log = logging.getLogger()
log.setLevel(logging.INFO)

sqs = boto3.client("sqs")
ssm = boto3.client("ssm")

QUEUE_URL = os.environ["INBOUND_QUEUE_URL"]
PARAM_PREFIX = os.environ.get("PARAM_PREFIX", "/mcb/whatsapp")

_cache: dict[str, str] = {}


def _param(name: str) -> str:
    """Read a parameter once per container.

    This Lambda is outside the VPC precisely so it can reach SSM and SQS
    without a NAT gateway - the webhook needs neither Redshift nor anything
    else private.
    """
    if name not in _cache:
        r = ssm.get_parameter(Name=f"{PARAM_PREFIX}/{name}",
                              WithDecryption=True)
        _cache[name] = r["Parameter"]["Value"]
    return _cache[name]


def _verified(raw: bytes, header: str | None) -> bool:
    """HMAC-SHA256 of the RAW body against the app secret.

    Raw, not re-serialised: `json.dumps(json.loads(body))` changes whitespace
    and key order, and the signature is over the bytes Meta actually sent.
    """
    if not header or not header.startswith("sha256="):
        return False
    expected = hmac.new(_param("app-secret").encode(), raw,
                        hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, header[len("sha256="):])


def _messages(payload: dict):
    """Yield the inbound text messages in a webhook payload.

    Meta delivers status callbacks (delivered, read) down the same webhook.
    Those are not messages and must not become agent turns.
    """
    for entry in payload.get("entry") or []:
        for change in entry.get("changes") or []:
            value = change.get("value") or {}
            meta = value.get("metadata") or {}
            contacts = {c.get("wa_id"): (c.get("profile") or {}).get("name")
                        for c in value.get("contacts") or []}
            for m in value.get("messages") or []:
                if m.get("type") != "text":
                    # Media arrives as an id that must be fetched with the
                    # token. Out of scope here; acknowledged, not dropped
                    # silently.
                    log.info("ignoring message type=%s", m.get("type"))
                    continue
                yield {
                    "channel": "whatsapp",
                    "message_id": m.get("id"),
                    "from": m.get("from"),
                    "display_name": contacts.get(m.get("from")),
                    "text": (m.get("text") or {}).get("body") or "",
                    "timestamp": m.get("timestamp"),
                    "phone_number_id": meta.get("phone_number_id"),
                }


def handler(event, context):                                 # noqa: ANN001
    method = ((event.get("requestContext") or {}).get("http") or {}).get("method")
    headers = {k.lower(): v for k, v in (event.get("headers") or {}).items()}

    if method == "GET":
        q = event.get("queryStringParameters") or {}
        if (q.get("hub.mode") == "subscribe"
                and q.get("hub.verify_token") == _param("verify-token")):
            log.info("subscription verified")
            return {"statusCode": 200, "body": q.get("hub.challenge", "")}
        log.warning("subscription verification failed")
        return {"statusCode": 403, "body": "forbidden"}

    raw = event.get("body") or ""
    if event.get("isBase64Encoded"):
        import base64

        raw_bytes = base64.b64decode(raw)
    else:
        raw_bytes = raw.encode()

    if not _verified(raw_bytes, headers.get("x-hub-signature-256")):
        # 403, not 200: this did not come from Meta, so there is nothing to
        # acknowledge.
        log.warning("signature verification failed")
        return {"statusCode": 403, "body": "bad signature"}

    try:
        payload = json.loads(raw_bytes or b"{}")
    except Exception:                                        # noqa: BLE001
        log.exception("unparseable body")
        return {"statusCode": 200, "body": "ignored"}

    sent = 0
    for msg in _messages(payload):
        sqs.send_message(
            QueueUrl=QUEUE_URL,
            MessageBody=json.dumps(msg),
            # One conversation is ordered; different customers are not
            # related. Grouping by sender keeps a person's turns in order
            # without serialising everybody behind one of them.
            MessageGroupId=msg["from"],
            # Meta retries deliver the same message id again. The queue
            # deduplicates it rather than the agent having to.
            MessageDeduplicationId=msg["message_id"],
        )
        sent += 1

    log.info("enqueued %d message(s)", sent)
    # Always 200 once the signature is good. A non-200 is a retry, and a
    # retry of something we chose not to act on is a loop.
    return {"statusCode": 200, "body": json.dumps({"queued": sent})}
