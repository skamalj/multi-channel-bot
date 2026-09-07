"""Channel-agnostic inbound event.

Every channel normalises to IngestEvent. The agent, resolver and tools never
learn which channel a message arrived on - which is what lets WhatsApp be
bolted on later without touching anything below this file.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal, Protocol

from pydantic import BaseModel, Field

Kind = Literal["text", "document", "image", "button", "unsupported"]


class MediaRef(BaseModel):
    media_id: str
    mime: str
    sha256: str | None = None
    filename: str | None = None
    size: int | None = None


class IngestEvent(BaseModel):
    """Normalised inbound message. ~1 KB - this is what crosses a queue."""

    message_id: str                     # dedupe key (wamid on WhatsApp)
    user_id: str                        # reply address AND resolver thread key
    channel: str                        # "webchat" | "whatsapp"
    channel_identity: str | None = None # which of OUR endpoints was addressed
    display_name: str | None = None
    ts: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    kind: Kind = "text"
    text: str | None = None             # body OR caption - normalised to one field
    button_id: str | None = None
    quoted_message_id: str | None = None
    media: MediaRef | None = None

    # Entry context: a deep link, QR or embed can pre-declare the route.
    entry_persona: str | None = None
    entry_lob: str | None = None


class OutboundMessage(BaseModel):
    user_id: str
    channel: str
    text: str
    buttons: list[dict[str, str]] = Field(default_factory=list)
    reply_to: str | None = None
    attachments: list[str] = Field(default_factory=list)


class ChannelAdapter(Protocol):
    name: str

    def parse(self, raw: dict) -> list[IngestEvent]:
        """Raw channel payload -> zero or more normalised events."""

    async def send(self, msg: OutboundMessage) -> str:
        """Deliver a reply. Returns the provider message id."""
