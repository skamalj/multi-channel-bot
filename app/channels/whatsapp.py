"""WhatsApp Cloud API adapter - parsing is real, sending is not yet wired.

Bolted on later: set WA_* env vars and register this adapter in main.py.
The parse() below is the full field mapping from the design doc, so the
only work left when the number arrives is the outbound HTTP call.
"""
from __future__ import annotations

from datetime import datetime, timezone

from .base import ChannelAdapter, IngestEvent, MediaRef, OutboundMessage


class WhatsAppAdapter(ChannelAdapter):
    name = "whatsapp"

    def parse(self, raw: dict) -> list[IngestEvent]:
        events: list[IngestEvent] = []
        for entry in raw.get("entry", []):
            for change in entry.get("changes", []):
                value = change.get("value", {})

                # Delivery receipts for messages WE sent. Counting them as
                # inbound messages is the classic first-week infinite loop.
                if "statuses" in value and "messages" not in value:
                    continue

                meta = value.get("metadata", {})
                contacts = {c["wa_id"]: c for c in value.get("contacts", [])}

                for m in value.get("messages", []):
                    wa_id = m["from"]
                    contact = contacts.get(wa_id, {})
                    kind, text, media = "unsupported", None, None

                    if m["type"] == "text":
                        kind, text = "text", m["text"]["body"]
                    elif m["type"] in ("document", "image"):
                        blob = m[m["type"]]
                        kind = m["type"]
                        text = blob.get("caption")
                        media = MediaRef(
                            media_id=blob["id"],
                            mime=blob.get("mime_type", ""),
                            sha256=blob.get("sha256"),
                            filename=blob.get("filename"),
                        )
                    elif m["type"] == "interactive":
                        kind = "button"

                    events.append(
                        IngestEvent(
                            message_id=m["id"],
                            user_id=wa_id,
                            channel=self.name,
                            channel_identity=meta.get("phone_number_id"),
                            display_name=contact.get("profile", {}).get("name"),
                            ts=datetime.fromtimestamp(int(m["timestamp"]), tz=timezone.utc),
                            kind=kind,
                            text=text,
                            button_id=(m.get("interactive", {})
                                        .get("button_reply", {}).get("id")),
                            quoted_message_id=m.get("context", {}).get("id"),
                            media=media,
                        )
                    )
        return events

    async def send(self, msg: OutboundMessage) -> str:  # pragma: no cover
        raise NotImplementedError(
            "WhatsApp outbound is bolted on separately. "
            "POST /{phone_number_id}/messages with to=user_id verbatim."
        )
