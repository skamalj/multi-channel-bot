"""Inbound queue -> the agent -> outbound queue.

The webhook is deliberately ignorant, so this is where a WhatsApp message
becomes an agent turn. It invokes the AgentCore Runtime over AG-UI exactly as
the console does, reads the event stream, and puts the answer on the outbound
queue for the formatter.

**The thread is the phone number.** That is the whole reason the agent's
memory works on WhatsApp at all: `threadId` is the customer's `wa_id`, so the
resolver thread and the bot thread are the same ones the console would use for
that person, and a conversation that started on the web continues on the
phone.

**Only the final answer crosses the queue.** The AG-UI stream carries steps,
tool calls and traces, and those are for the glass box, not for a customer's
phone. This reads the whole stream, keeps the TEXT_MESSAGE_CONTENT, and drops
the rest - the process is observable, the message is what gets sent.

A failure here is a retry, not a lost message: the SQS message stays until
this returns cleanly, and after `maxReceiveCount` it lands in the DLQ where
somebody can look at it.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os

import boto3

log = logging.getLogger()
log.setLevel(logging.INFO)

agentcore = boto3.client("bedrock-agentcore")
sqs = boto3.client("sqs")

RUNTIME_ARN = os.environ["AGENT_RUNTIME_ARN"]
QUALIFIER = os.environ.get("AGENT_QUALIFIER", "live")
OUTBOUND_QUEUE_URL = os.environ["OUTBOUND_QUEUE_URL"]


def _session_id(wa_id: str) -> str:
    """AgentCore requires at least 33 characters; a phone number is 12.

    Derived rather than padded so the same customer always lands on the same
    runtime session.
    """
    return hashlib.sha256(f"whatsapp:{wa_id}".encode()).hexdigest()


def _answer(stream) -> tuple[str, list[str]]:
    """The customer-facing text out of an AG-UI event stream.

    Returns the answer and the tool names seen, because the formatter can say
    something useful about a reply that came from a policy lookup that it
    cannot about one that did not.
    """
    text_parts: list[str] = []
    tools: list[str] = []
    error: str | None = None

    for raw in stream.iter_lines() if hasattr(stream, "iter_lines") else stream:
        line = raw.decode() if isinstance(raw, (bytes, bytearray)) else str(raw)
        line = line.strip()
        if not line.startswith("data:"):
            continue
        try:
            ev = json.loads(line[5:].strip())
        except Exception:                                    # noqa: BLE001
            continue
        kind = ev.get("type")
        if kind == "TEXT_MESSAGE_CONTENT":
            text_parts.append(ev.get("delta") or "")
        elif kind == "TOOL_CALL_START":
            tools.append(ev.get("toolCallName") or "")
        elif kind == "RUN_ERROR":
            error = f"{ev.get('code')}: {ev.get('message')}"

    if error and not text_parts:
        raise RuntimeError(f"agent run failed - {error}")
    return "".join(text_parts).strip(), [t for t in tools if t]


def _invoke(msg: dict) -> tuple[str, list[str]]:
    payload = {
        "threadId": msg["from"],          # the phone number IS the thread
        "runId": msg["message_id"],
        "messages": [{"id": msg["message_id"], "role": "user",
                      "content": msg.get("text") or ""}],
        "state": {}, "tools": [], "context": [],
        "forwardedProps": {
            "channel": "whatsapp",
            "channel_identity": msg["from"],
            "display_name": msg.get("display_name"),
        },
    }
    resp = agentcore.invoke_agent_runtime(
        agentRuntimeArn=RUNTIME_ARN,
        runtimeSessionId=_session_id(msg["from"]),
        qualifier=QUALIFIER,
        payload=json.dumps(payload).encode())
    return _answer(resp["response"])


def handler(event, context):                                 # noqa: ANN001
    """One SQS batch. Failures are reported per message, not per batch.

    `ReportBatchItemFailures` matters here: without it one bad message would
    make the whole batch retry, and a customer who already got an answer
    would get it again.
    """
    failures: list[dict] = []

    for record in event.get("Records") or []:
        try:
            msg = json.loads(record["body"])
            log.info("turn for %s (%s chars)", msg.get("from"),
                     len(msg.get("text") or ""))
            answer, tools = _invoke(msg)
            if not answer:
                # The agent ran and said nothing. That is a real outcome -
                # a suppressed contact, for instance - and not an error to
                # retry.
                log.info("no reply produced for %s", msg.get("from"))
                continue
            sqs.send_message(
                QueueUrl=OUTBOUND_QUEUE_URL,
                MessageBody=json.dumps({
                    "to": msg["from"],
                    "display_name": msg.get("display_name"),
                    "in_reply_to": msg["message_id"],
                    "phone_number_id": msg.get("phone_number_id"),
                    "text": answer,
                    "tools": tools,
                }),
                MessageGroupId=msg["from"],
                MessageDeduplicationId=f"{msg['message_id']}-reply",
            )
        except Exception as exc:                             # noqa: BLE001
            log.exception("message failed")
            failures.append({"itemIdentifier": record["messageId"]})

    return {"batchItemFailures": failures}
