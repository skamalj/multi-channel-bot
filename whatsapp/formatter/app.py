"""Outbound queue -> a message shaped for a phone -> WhatsApp.

The agent writes for a screen. WhatsApp is a different medium: no tables, no
markdown headings, a much shorter attention span, and a 4096-character hard
limit. This is the second agent in the chain and its only job is presentation.

**It may not change what was said.** That constraint is the whole design.
The answer it receives has already been through the citation guardrail and
Bedrock's contextual grounding; if this were allowed to rewrite freely it
would be a second, ungoverned author sitting after every control the system
has. So the prompt forbids adding facts, and the output is checked against
the input: every number in the formatted message must appear in the original.
If it does not, the original is sent instead.

That check is cheap and blunt and it is the right shape - a formatter that
invents a premium is exactly the failure mode worth spending code on, and a
formatter that merely fails to be pretty is not.
"""
from __future__ import annotations

import json
import logging
import os
import re
import urllib.error
import urllib.request

import boto3

log = logging.getLogger()
log.setLevel(logging.INFO)

bedrock = boto3.client("bedrock-runtime")
ssm = boto3.client("ssm")

MODEL_ID = os.environ.get("FORMATTER_MODEL_ID", "apac.amazon.nova-lite-v1:0")
GRAPH_VERSION = os.environ.get("GRAPH_VERSION", "v21.0")
PARAM_PREFIX = os.environ.get("PARAM_PREFIX", "/mcb/whatsapp")
GUARDRAIL_ID = os.environ.get("GUARDRAIL_ID", "")
GUARDRAIL_VERSION = os.environ.get("GUARDRAIL_VERSION", "DRAFT")
WHATSAPP_LIMIT = 4096

_cache: dict[str, str] = {}

SYSTEM = (
    "You reformat an insurance assistant's reply for WhatsApp.\n"
    "RULES, in order of importance:\n"
    "1. Never add, remove or change a fact, a number, an amount, a date, a "
    "policy number or a name. You are not answering; you are typesetting.\n"
    "2. No markdown headings, no tables, no bullet characters other than a "
    "leading dash. WhatsApp renders *bold* with single asterisks.\n"
    "3. Short lines. Blank line between ideas. Under 900 characters if the "
    "content allows it.\n"
    "4. Keep the closing question if there is one - it is how the "
    "conversation continues.\n"
    "Return only the reformatted message."
)

# A number that carries meaning: money, percentages, policy and claim
# references. Used to prove the formatter invented nothing.
_FIGURE = re.compile(r"\d[\d,]*\.?\d*%?|\b[A-Z]{2,5}-\d{4,}\b")


def _param(name: str) -> str:
    if name not in _cache:
        r = ssm.get_parameter(Name=f"{PARAM_PREFIX}/{name}",
                              WithDecryption=True)
        _cache[name] = r["Parameter"]["Value"]
    return _cache[name]


def _figures(text: str) -> set[str]:
    return {m.group(0).rstrip(".").replace(",", "")
            for m in _FIGURE.finditer(text or "")}


def _format(text: str) -> str:
    """Ask the small model to typeset. Falls back to the original, always.

    Fails CLOSED in the sense that matters: any doubt and the customer gets
    the governed text rather than a prettier unverified one.
    """
    body = {
        "messages": [{"role": "user",
                      "content": [{"text": f"{SYSTEM}\n\n---\n{text}"}]}],
        "inferenceConfig": {"temperature": 0.0, "maxTokens": 1200},
    }
    kwargs = {"modelId": MODEL_ID, "body": json.dumps(body),
              "contentType": "application/json"}
    if GUARDRAIL_ID:
        kwargs["guardrailIdentifier"] = GUARDRAIL_ID
        kwargs["guardrailVersion"] = GUARDRAIL_VERSION

    try:
        resp = bedrock.invoke_model(**kwargs)
        payload = json.loads(resp["body"].read())
        out = ""
        for block in (payload.get("output", {}).get("message", {})
                      .get("content") or []):
            out += block.get("text", "")
        out = out.strip()
    except Exception as exc:                                 # noqa: BLE001
        log.warning("formatter model failed: %s", type(exc).__name__)
        return text

    if not out:
        return text

    # The check that makes this safe to run after the guardrails: a figure in
    # the formatted message that was not in the original means the formatter
    # authored something, and it does not get to do that.
    invented = _figures(out) - _figures(text)
    if invented:
        log.warning("formatter introduced figures %s - sending the original",
                    sorted(invented))
        return text

    return out[:WHATSAPP_LIMIT]


def _send(phone_number_id: str, to: str, text: str) -> dict:
    url = (f"https://graph.facebook.com/{GRAPH_VERSION}/"
           f"{phone_number_id or _param('phone-number-id')}/messages")
    payload = json.dumps({
        "messaging_product": "whatsapp",
        "recipient_type": "individual",
        "to": to,
        "type": "text",
        "text": {"preview_url": False, "body": text[:WHATSAPP_LIMIT]},
    }).encode()
    req = urllib.request.Request(
        url, data=payload, method="POST",
        headers={"Authorization": f"Bearer {_param('access-token')}",
                 "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=20) as r:
        return json.loads(r.read() or b"{}")


def handler(event, context):                                 # noqa: ANN001
    failures: list[dict] = []

    for record in event.get("Records") or []:
        try:
            msg = json.loads(record["body"])
            original = msg.get("text") or ""
            if not original:
                continue
            pretty = _format(original)
            log.info("to=%s %d chars -> %d", msg.get("to"),
                     len(original), len(pretty))
            result = _send(msg.get("phone_number_id"), msg["to"], pretty)
            log.info("sent %s", (result.get("messages") or [{}])[0].get("id"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode(errors="replace")[:400]
            log.error("graph API %s: %s", exc.code, detail)
            # 4xx will not succeed on a retry; 5xx might.
            if exc.code >= 500:
                failures.append({"itemIdentifier": record["messageId"]})
        except Exception:                                    # noqa: BLE001
            log.exception("send failed")
            failures.append({"itemIdentifier": record["messageId"]})

    return {"batchItemFailures": failures}
