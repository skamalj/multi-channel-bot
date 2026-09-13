"""Meta's WhatsApp webhook. Verifies, forwards, and nothing else.

This is the only publicly reachable thing in the whole build, so it does the
least possible: it proves the request came from Meta, drops the RAW payload
onto a queue, and returns 200. No parsing, no model, no database. Parsing,
media fetch and normalisation happen in the processor Lambda behind the queue,
where latency does not matter - a slow or failing webhook makes Meta retry and
eventually disable the subscription, so the work belongs behind the queue.

**The signature check is not optional.** A Lambda Function URL with
`AuthType: NONE` accepts a request from anyone who finds the address. Without
verifying `X-Hub-Signature-256` this endpoint would let a stranger inject
messages the agent would answer as though they came from a customer.

Two paths, both required by Meta:

  GET  - subscription handshake, echo `hub.challenge` if the verify token
         matches.
  POST - message delivery. The whole validated body is forwarded to the raw
         queue verbatim (the signature is over those exact bytes, and the
         processor re-parses them). Always 200 after a good signature: a
         non-200 is a retry, and a retry of something we cannot act on is a
         loop.
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

# The RAW queue the processor consumes. Standard (not FIFO): the webhook has no
# single message id to key a whole payload on, and Meta retries are deduped
# downstream where the processor enqueues per-message to the FIFO inbound queue.
RAW_QUEUE_URL = os.environ["RAW_QUEUE_URL"]
PARAM_PREFIX = os.environ.get("PARAM_PREFIX", "/mcb/whatsapp")

_cache: dict[str, str] = {}


def _param(name: str) -> str:
    """Read a parameter once per container. This Lambda is outside the VPC so
    it can reach SSM and SQS without a NAT gateway."""
    if name not in _cache:
        r = ssm.get_parameter(Name=f"{PARAM_PREFIX}/{name}",
                              WithDecryption=True)
        _cache[name] = r["Parameter"]["Value"]
    return _cache[name]


def _verified(raw: bytes, header: str | None) -> bool:
    """HMAC-SHA256 of the RAW body against the app secret. Raw, not
    re-serialised: the signature is over the bytes Meta actually sent."""
    if not header or not header.startswith("sha256="):
        return False
    expected = hmac.new(_param("app-secret").encode(), raw,
                        hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, header[len("sha256="):])


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

    # Forward the validated raw payload as-is. The processor parses it, drops
    # status callbacks, fetches media, and normalises per message.
    sqs.send_message(QueueUrl=RAW_QUEUE_URL,
                     MessageBody=raw_bytes.decode("utf-8", "replace"))
    log.info("forwarded payload to raw queue")
    return {"statusCode": 200, "body": json.dumps({"forwarded": True})}
