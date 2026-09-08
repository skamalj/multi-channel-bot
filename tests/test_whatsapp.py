"""The WhatsApp path, tested where it can be tested without Meta.

Three things in this chain are pure functions of their input, and they are
exactly the three worth getting right:

* the webhook's signature check - the only thing standing between a public
  URL and a stranger putting words in a customer's mouth;
* the parse that decides what counts as a message, because Meta sends
  delivery receipts down the same webhook and they must not become turns;
* the formatter's figure check, which is what stops a presentation model
  becoming a second, ungoverned author after every guardrail in the system.

The send itself and the agent invocation need real credentials and are not
faked here - a mock of the Graph API would only assert that the mock works.
"""
from __future__ import annotations

import hashlib
import hmac
import importlib.util
import json
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _load(name: str, relpath: str, env: dict):
    """Load a Lambda handler module by path, with its environment in place."""
    import os

    for k, v in env.items():
        os.environ[k] = v
    spec = importlib.util.spec_from_file_location(name, ROOT / relpath)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def webhook():
    boto3 = pytest.importorskip("boto3")
    mod = _load("wa_webhook", "whatsapp/webhook/app.py",
                {"INBOUND_QUEUE_URL": "https://example.invalid/q",
                 "PARAM_PREFIX": "/mcb/whatsapp",
                 # These modules build their boto3 clients at import, which is
                 # right for a Lambda - the client is reused across warm
                 # invocations - but it means a region has to exist even when
                 # nothing is called.
                 "AWS_DEFAULT_REGION": "ap-south-1"})
    # The parameter store is not reachable in a unit test, and these are the
    # values it would return.
    mod._cache.update({"app-secret": "s3cr3t", "verify-token": "hello-meta"})
    return mod


@pytest.fixture(scope="module")
def formatter():
    pytest.importorskip("boto3")
    return _load("wa_formatter", "whatsapp/formatter/app.py",
                 {"PARAM_PREFIX": "/mcb/whatsapp",
                  "AWS_DEFAULT_REGION": "ap-south-1"})


def _signed(body: dict, secret: str = "s3cr3t") -> tuple[bytes, str]:
    raw = json.dumps(body).encode()
    sig = hmac.new(secret.encode(), raw, hashlib.sha256).hexdigest()
    return raw, f"sha256={sig}"


# --- signature -------------------------------------------------------------
def test_a_correctly_signed_body_is_accepted(webhook):
    raw, sig = _signed({"entry": []})
    assert webhook._verified(raw, sig)


def test_an_unsigned_body_is_rejected(webhook):
    raw, _ = _signed({"entry": []})
    assert not webhook._verified(raw, None)
    assert not webhook._verified(raw, "")


def test_a_body_signed_with_the_wrong_secret_is_rejected(webhook):
    raw, sig = _signed({"entry": []}, secret="not-the-secret")
    assert not webhook._verified(raw, sig)


def test_a_tampered_body_is_rejected(webhook):
    """The signature is over the bytes, so changing one invalidates it."""
    raw, sig = _signed({"entry": [], "amount": 100})
    tampered = raw.replace(b"100", b"900")
    assert not webhook._verified(tampered, sig)


def test_an_unsigned_post_is_refused_end_to_end(webhook):
    raw, _ = _signed({"entry": []})
    resp = webhook.handler(
        {"requestContext": {"http": {"method": "POST"}},
         "headers": {}, "body": raw.decode()}, None)
    assert resp["statusCode"] == 403


# --- subscription handshake ------------------------------------------------
def test_the_handshake_echoes_the_challenge_for_the_right_token(webhook):
    resp = webhook.handler(
        {"requestContext": {"http": {"method": "GET"}},
         "queryStringParameters": {"hub.mode": "subscribe",
                                   "hub.verify_token": "hello-meta",
                                   "hub.challenge": "echo-me"}}, None)
    assert resp["statusCode"] == 200
    assert resp["body"] == "echo-me"


def test_the_handshake_refuses_the_wrong_token(webhook):
    resp = webhook.handler(
        {"requestContext": {"http": {"method": "GET"}},
         "queryStringParameters": {"hub.mode": "subscribe",
                                   "hub.verify_token": "wrong",
                                   "hub.challenge": "echo-me"}}, None)
    assert resp["statusCode"] == 403


# --- what counts as a message ---------------------------------------------
def _payload(messages: list[dict]) -> dict:
    return {"entry": [{"changes": [{"value": {
        "metadata": {"phone_number_id": "PN1"},
        "contacts": [{"wa_id": "919820000009",
                      "profile": {"name": "Priya Sharma"}}],
        "messages": messages}}]}]}


def test_a_text_message_becomes_one_turn(webhook):
    out = list(webhook._messages(_payload([
        {"id": "wamid.1", "from": "919820000009", "type": "text",
         "timestamp": "1", "text": {"body": "what is my cover"}}])))
    assert len(out) == 1
    assert out[0]["text"] == "what is my cover"
    assert out[0]["from"] == "919820000009"
    assert out[0]["display_name"] == "Priya Sharma"
    assert out[0]["phone_number_id"] == "PN1"


def test_a_delivery_receipt_is_not_a_turn(webhook):
    """Meta sends statuses down the same webhook. They are not messages."""
    payload = {"entry": [{"changes": [{"value": {
        "metadata": {"phone_number_id": "PN1"},
        "statuses": [{"id": "wamid.1", "status": "delivered"}]}}]}]}
    assert list(webhook._messages(payload)) == []


def test_an_image_is_not_treated_as_text(webhook):
    """Media needs a fetch with the token; it is skipped, not half-handled."""
    out = list(webhook._messages(_payload([
        {"id": "wamid.2", "from": "919820000009", "type": "image",
         "image": {"id": "media-1"}}])))
    assert out == []


# --- the formatter may not invent ------------------------------------------
def test_figures_are_extracted_from_money_percentages_and_references(formatter):
    figs = formatter._figures(
        "Premium is 12,499 including 18% GST on policy PHS-4471902.")
    assert "12499" in figs
    assert "18%" in figs
    assert "PHS-4471902" in figs


def test_a_reformat_that_keeps_every_figure_is_accepted(formatter, monkeypatch):
    original = "Your premium is 12,499 and the waiting period is 36 months."
    pretty = "*Premium*: 12,499\n\nWaiting period: 36 months."
    monkeypatch.setattr(formatter, "_format", lambda t: pretty)
    assert not (formatter._figures(pretty) - formatter._figures(original))


def test_a_reformat_that_invents_a_figure_is_caught(formatter):
    """The check that makes a presentation model safe to run last."""
    original = "Your premium is 12,499."
    invented = "Your premium is 12,499, with 25% no-claim bonus."
    assert formatter._figures(invented) - formatter._figures(original) == {"25%"}


def test_a_reformat_that_changes_a_figure_is_caught(formatter):
    original = "The waiting period is 36 months."
    changed = "The waiting period is 9 months."
    assert formatter._figures(changed) - formatter._figures(original) == {"9"}
