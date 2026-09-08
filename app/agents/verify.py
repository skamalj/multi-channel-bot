"""Is this answer supported by the passages it was given?

This replaces a word list that tried to detect "a claim about cover" by
looking for the word "cover". That approach failed in both directions on the
same turn: it refused the bot's own question - "Ages of family members you
want to cover" - and it let "The premium is 12,499" through unchecked,
because a comma defeated the number regex. Wrong about a question, silent
about a price.

Deciding whether a sentence asserts something, and whether a passage supports
it, is a language problem. A regex cannot tell an assertion from a question,
a quotation, or the customer's own words repeated back. A small model can.

**What this does NOT do.** It never decides whether a citation is valid. The
model is handed passages numbered [1]..[k] and code checks the number against
what was actually sent - array membership, not judgement. Asking a model "is
[2] one of these four passages" would invite it to hallucinate the one thing
that has to be certain.

**Failure is loud, and degrades rather than disappears.** If this is
unavailable the turn proceeds, the trace records that verification did not
run, and the citation-resolution rule still applies. That is weaker, and
saying so is better than either silently failing open or refusing every turn
because a model endpoint was slow.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field

from app.config import settings

log = logging.getLogger("mcb.verify")

SYSTEM = """You check an insurance assistant's draft reply against the source \
passages it was given. You are not writing the reply and you are not \
improving it.

A claim is UNSUPPORTED when the draft states something about the product - \
cover, an exclusion, an amount, a waiting period, a premium, a status or a \
decision - that the passages do not say.

These are NOT claims, and must never be reported:
- a question to the customer, or a request for their details
- repeating back what the customer just said
- describing what you are about to do ("I can look that up")
- a general statement carrying no specific commitment
- a fact listed under ESTABLISHED below, which was confirmed earlier in this \
same conversation

Reply with JSON only:
{"unsupported": ["<exact sentence>", ...]}
An empty list means everything checks out. Quote sentences exactly as they \
appear in the draft."""


@dataclass(frozen=True)
class Verdict:
    ran: bool
    unsupported: list[str] = field(default_factory=list)
    error: str | None = None

    @property
    def ok(self) -> bool:
        return not self.unsupported


def configured() -> bool:
    import os

    cfg = settings()
    if os.getenv("NO_AWS") == "1" or cfg.no_aws or cfg.mock_llm:
        return False
    return bool(cfg.verifier_enabled)


def _passages(chunks: list[dict]) -> str:
    out = []
    for c in chunks:
        ref = c.get("ref")
        if ref is None:
            continue
        text = str(c.get("text") or "")[:1500]
        src = c.get("source") or c.get("doc_id") or ""
        out.append(f"[{ref}] ({src})\n{text}")
    return "\n\n".join(out)


def _json_from(text: str) -> dict:
    """Take the first object. Models wrap JSON in prose and fences."""
    m = re.search(r"\{.*\}", text or "", re.S)
    if not m:
        return {}
    try:
        return json.loads(m.group(0))
    except Exception:                                        # noqa: BLE001
        return {}


def check(answer: str, chunks: list[dict],
          established: list[str] | None = None) -> Verdict:
    """Which sentences in `answer` the passages do not support.

    `established` is what this conversation already confirmed with a source.
    Without it a follow-up - "how many months was that again?" - gets refused
    for restating something the customer was correctly told a moment ago,
    which is how a bot ends up unable to hold a conversation.
    """
    if not configured() or not answer.strip():
        return Verdict(ran=False)

    from langchain_core.messages import HumanMessage, SystemMessage

    from app.llm.bedrock import get_llm

    body = [f"PASSAGES\n{_passages(chunks) or '(none were retrieved)'}"]
    if established:
        body.append("ESTABLISHED EARLIER IN THIS CONVERSATION\n"
                    + "\n".join(f"- {e}" for e in established[-12:]))
    body.append(f"DRAFT REPLY\n{answer}")

    try:
        resp = get_llm(small=True).invoke([
            SystemMessage(content=SYSTEM),
            HumanMessage(content="\n\n".join(body)[:24000])])
        content = resp.content
        if isinstance(content, list):          # Converse content blocks
            content = " ".join(b.get("text", "") for b in content
                               if isinstance(b, dict))
    except Exception as exc:                                 # noqa: BLE001
        log.warning("verifier unavailable: %s: %s", type(exc).__name__, exc)
        return Verdict(ran=False, error=type(exc).__name__)

    data = _json_from(str(content))
    raw = data.get("unsupported")
    if not isinstance(raw, list):
        # A malformed verdict is not a pass. It is a failed check, and the
        # trace should say so rather than imply the answer was cleared.
        return Verdict(ran=False, error="unparseable_verdict")

    return Verdict(ran=True,
                   unsupported=[str(x) for x in raw if str(x).strip()])
