"""Processor: turn a raw WhatsApp webhook payload into normalised agent turns.

Sits behind the webhook's raw queue so all the slow, fallible work is off the
millisecond path Meta measures. Per payload it:

  * drops status callbacks (delivered/read are not turns);
  * fans the payload into one normalised message per inbound message;
  * for media (image/document/audio/video/voice/sticker), fetches the bytes
    from the Graph API and persists them to S3, injecting only the METADATA
    (s3 key, mime, sha256, size) into the message - never the bytes;
  * enqueues each normalised message to the FIFO inbound queue the agent path
    consumes, keyed `MessageGroupId=from`, `MessageDeduplicationId=message_id`
    (so a Meta retry of the same payload is deduped here, and one person's
    turns stay ordered without serialising everyone).

Self-contained (stdlib + boto3): it does not import the `app` package, so it
stays a light zip Lambda. The parse mirrors `app.channels.whatsapp` closely;
if the two drift, that adapter is the source of truth to reconcile against.
"""
from __future__ import annotations

import hashlib
import json
import logging
import mimetypes
import os
import urllib.request

import boto3

log = logging.getLogger()
log.setLevel(logging.INFO)

sqs = boto3.client("sqs")
s3 = boto3.client("s3")
ssm = boto3.client("ssm")

INBOUND_QUEUE_URL = os.environ["INBOUND_QUEUE_URL"]
DOCS_BUCKET = os.environ["DOCS_BUCKET"]
PARAM_PREFIX = os.environ.get("PARAM_PREFIX", "/mcb/whatsapp")
GRAPH_VERSION = os.environ.get("GRAPH_VERSION", "v20.0")

# Media kinds carry an id to fetch; everything else is text-shaped or ignored.
_MEDIA_TYPES = {"image", "document", "audio", "video", "voice", "sticker"}

_cache: dict[str, str] = {}


def _param(name: str) -> str:
    if name not in _cache:
        r = ssm.get_parameter(Name=f"{PARAM_PREFIX}/{name}", WithDecryption=True)
        _cache[name] = r["Parameter"]["Value"]
    return _cache[name]


def _graph_get(url: str) -> bytes:
    req = urllib.request.Request(
        url, headers={"Authorization": f"Bearer {_param('access-token')}"})
    with urllib.request.urlopen(req, timeout=20) as resp:      # noqa: S310
        return resp.read()


def _store_media(media_id: str, declared_mime: str | None,
                 filename: str | None) -> dict | None:
    """Graph media-id -> URL -> bytes -> S3. Returns the metadata to inject, or
    None if the fetch failed (the turn still goes through, sans attachment)."""
    try:
        meta = json.loads(_graph_get(
            f"https://graph.facebook.com/{GRAPH_VERSION}/{media_id}"))
        data = _graph_get(meta["url"])
    except Exception as exc:                                   # noqa: BLE001
        log.warning("media fetch failed id=%s: %s", media_id, type(exc).__name__)
        return None
    mime = meta.get("mime_type") or declared_mime or "application/octet-stream"
    sha = hashlib.sha256(data).hexdigest()
    ext = (mimetypes.guess_extension(mime.split(";")[0]) or "").lstrip(".")
    key = f"documents/whatsapp/{sha}" + (f".{ext}" if ext else "")
    s3.put_object(Bucket=DOCS_BUCKET, Key=key, Body=data, ContentType=mime)
    return {"media_id": media_id, "mime": mime, "filename": filename,
            "sha256": sha, "size": len(data),
            "s3_bucket": DOCS_BUCKET, "s3_key": key}


def _normalise(payload: dict):
    """Yield one normalised message dict per inbound message; drop statuses."""
    for entry in payload.get("entry") or []:
        for change in entry.get("changes") or []:
            value = change.get("value") or {}
            if "statuses" in value and "messages" not in value:
                continue                                       # delivery receipt
            meta = value.get("metadata") or {}
            phone_number_id = meta.get("phone_number_id")
            contacts = {c.get("wa_id"): (c.get("profile") or {}).get("name")
                        for c in value.get("contacts") or []}
            for m in value.get("messages") or []:
                yield _one(m, contacts, phone_number_id)


def _one(m: dict, contacts: dict, phone_number_id: str | None) -> dict:
    mtype = m.get("type")
    frm = m.get("from")
    out = {"channel": "whatsapp", "message_id": m.get("id"), "from": frm,
           "display_name": contacts.get(frm), "timestamp": m.get("timestamp"),
           "phone_number_id": phone_number_id, "kind": "text",
           "text": "", "media": None}

    if mtype == "text":
        out["text"] = (m.get("text") or {}).get("body") or ""
    elif mtype == "interactive":
        inter = m.get("interactive") or {}
        reply = inter.get("button_reply") or inter.get("list_reply") or {}
        out["kind"] = "button"
        out["button_id"] = reply.get("id")
        out["text"] = reply.get("title") or reply.get("id") or ""
    elif mtype == "button":                                    # template quick-reply
        out["kind"] = "button"
        out["button_id"] = (m.get("button") or {}).get("payload")
        out["text"] = (m.get("button") or {}).get("text") or ""
    elif mtype in _MEDIA_TYPES:
        media = m.get(mtype) or {}
        out["kind"] = "image" if mtype == "image" else "document"
        out["text"] = media.get("caption") or ""
        ref = _store_media(media.get("id"), media.get("mime_type"),
                           media.get("filename"))
        out["media"] = ref
        if ref is None:
            out["text"] = (out["text"] + " [attachment could not be fetched]").strip()
    else:
        out["kind"] = "unsupported"
        out["text"] = f"[unsupported message type: {mtype}]"

    return out


def handler(event, context):                                 # noqa: ANN001
    failures: list[dict] = []
    for record in event.get("Records") or []:
        try:
            payload = json.loads(record["body"])
            if payload.get("object") != "whatsapp_business_account":
                continue
            for msg in _normalise(payload):
                if not msg.get("message_id") or not msg.get("from"):
                    continue
                sqs.send_message(
                    QueueUrl=INBOUND_QUEUE_URL,
                    MessageBody=json.dumps(msg),
                    MessageGroupId=msg["from"],
                    MessageDeduplicationId=msg["message_id"])
        except Exception:                                      # noqa: BLE001
            log.exception("processing record failed")
            failures.append({"itemIdentifier": record.get("messageId")})
    return {"batchItemFailures": failures}
