"""Confirmation and idempotency for mutating tools (AG-6).

There is no interrupt here, and there could not be. On Lambda the process
ends when the reply is sent and hours pass before the next message, so
"waiting for a confirmation" is **a field that is still empty**, not a parked
node. The pending call lives in the bot session and the next turn either
finds an affirmative answer or it does not.

The idempotency key is the confirmation token, deliberately. Keying on the
message id would give the confirming turn a different key from the proposing
turn, which is precisely the retry that must not double-charge. The token
identifies the OPERATION; it survives the round trip; a replay of the same
token returns the first result rather than performing the write again.
"""
from __future__ import annotations

import hashlib
import json
import time
from typing import Any

# Nothing else lives here any more.
#
# park / pending_of / clear / read_answer / summarise are gone with the
# machinery that used them: a gate that intercepted a mutating call, wrote
# "I am about to create a motor quote - ncb pct: 20" into the conversation in
# the bot's voice, held the arguments in checkpointed state, and read the
# next message as nothing but an answer to that question. Asking is the
# model's job, told by the prompt and by the tool descriptions.
#
# The token stays, because it is the one part a sentence cannot do: it is the
# idempotency key, so the same call agreed to twice is written once.


def token_for(tool_name: str, args: dict[str, Any]) -> str:
    """Stable across the round trip: same tool, same arguments, same token."""
    body = json.dumps({"t": tool_name, "a": args}, sort_keys=True, default=str)
    return hashlib.sha256(body.encode("utf-8")).hexdigest()[:16]
