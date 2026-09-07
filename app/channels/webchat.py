"""Web chat adapter - the test harness channel.

Replies are returned in the HTTP response rather than pushed, so `send`
buffers them for the request handler to collect.
"""
from __future__ import annotations

import uuid
from collections import defaultdict

from .base import ChannelAdapter, IngestEvent, OutboundMessage


class WebChatAdapter(ChannelAdapter):
    name = "webchat"

    def __init__(self) -> None:
        self._outbox: dict[str, list[OutboundMessage]] = defaultdict(list)

    def parse(self, raw: dict) -> list[IngestEvent]:
        return [
            IngestEvent(
                message_id=raw.get("message_id") or uuid.uuid4().hex,
                user_id=raw["user_id"],
                channel=self.name,
                channel_identity="webchat",
                display_name=raw.get("display_name"),
                kind="text",
                text=raw.get("text"),
                entry_persona=raw.get("persona"),
                entry_lob=raw.get("lob"),
            )
        ]

    async def send(self, msg: OutboundMessage) -> str:
        mid = uuid.uuid4().hex
        self._outbox[msg.user_id].append(msg)
        return mid

    def drain(self, user_id: str) -> list[OutboundMessage]:
        out = self._outbox.pop(user_id, [])
        return out
