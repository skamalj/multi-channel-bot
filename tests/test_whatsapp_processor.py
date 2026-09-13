"""The WhatsApp processor's parse/normalise fan-out, offline.

The processor is a self-contained Lambda (stdlib + boto3); these exercise its
payload parsing without AWS - status callbacks dropped, one normalised message
per inbound message, media metadata injected (never bytes), and a media fetch
failure that still lets the turn through.
"""
import os

os.environ.setdefault("AWS_DEFAULT_REGION", "ap-south-1")   # Lambda sets this
os.environ.setdefault("INBOUND_QUEUE_URL", "http://queue")
os.environ.setdefault("DOCS_BUCKET", "mcb-documents-test")
os.environ.setdefault("PARAM_PREFIX", "/mcb/whatsapp")

from whatsapp.processor import app as proc  # noqa: E402


def _payload(message: dict, *, contacts=True) -> dict:
    return {"object": "whatsapp_business_account", "entry": [{"changes": [{
        "value": {
            "metadata": {"phone_number_id": "PN1"},
            "contacts": ([{"wa_id": "9199", "profile": {"name": "Asha"}}]
                         if contacts else []),
            "messages": [message]},
    }]}]}


def _text(**over):
    m = {"id": "wamid.1", "from": "9199", "type": "text", "timestamp": "1",
         "text": {"body": "hi"}}
    m.update(over)
    return m


def test_text_message_normalises():
    m = list(proc._normalise(_payload(_text())))[0]
    assert m["kind"] == "text" and m["text"] == "hi"
    assert m["from"] == "9199" and m["display_name"] == "Asha"
    assert m["phone_number_id"] == "PN1" and m["media"] is None
    assert m["message_id"] == "wamid.1"


def test_status_callbacks_are_dropped():
    payload = {"object": "whatsapp_business_account", "entry": [{"changes": [
        {"value": {"statuses": [{"status": "delivered", "id": "wamid.x"}]}}]}]}
    assert list(proc._normalise(payload)) == []


def test_interactive_button_reply():
    m = list(proc._normalise(_payload({
        "id": "wamid.2", "from": "9199", "type": "interactive", "timestamp": "1",
        "interactive": {"type": "button_reply",
                        "button_reply": {"id": "yes", "title": "Yes, go ahead"}},
    })))[0]
    assert m["kind"] == "button" and m["button_id"] == "yes"
    assert m["text"] == "Yes, go ahead"


def test_unsupported_type_is_flagged_not_dropped():
    m = list(proc._normalise(_payload({
        "id": "wamid.3", "from": "9199", "type": "location", "timestamp": "1",
        "location": {"latitude": 1, "longitude": 2}})))[0]
    assert m["kind"] == "unsupported" and "location" in m["text"]


def test_media_message_injects_metadata_not_bytes(monkeypatch):
    monkeypatch.setattr(proc, "_store_media", lambda *a: {
        "media_id": "M1", "mime": "image/jpeg", "filename": None,
        "sha256": "abc123", "size": 2048,
        "s3_bucket": "mcb-documents-test", "s3_key": "documents/whatsapp/abc123.jpeg"})
    m = list(proc._normalise(_payload({
        "id": "wamid.4", "from": "9199", "type": "image", "timestamp": "1",
        "image": {"id": "M1", "mime_type": "image/jpeg", "caption": "my car"}})))[0]
    assert m["kind"] == "image" and m["text"] == "my car"
    assert m["media"]["s3_key"] == "documents/whatsapp/abc123.jpeg"
    assert m["media"]["sha256"] == "abc123" and "s3_bucket" in m["media"]
    # No bytes ever ride on the message.
    assert "body" not in m["media"] and "content" not in m["media"]


def test_a_failed_media_fetch_still_yields_the_turn(monkeypatch):
    monkeypatch.setattr(proc, "_store_media", lambda *a: None)
    m = list(proc._normalise(_payload({
        "id": "wamid.5", "from": "9199", "type": "document", "timestamp": "1",
        "document": {"id": "M2", "mime_type": "application/pdf",
                     "filename": "policy.pdf", "caption": ""}})))[0]
    assert m["media"] is None
    assert "could not be fetched" in m["text"]
