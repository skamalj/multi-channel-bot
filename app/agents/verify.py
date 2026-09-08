"""Is this answer supported by the passages it was given?

This replaces a word list that tried to detect "a claim about cover" by
looking for the word "cover". That approach failed in both directions on the
same turn: it refused the bot's own question - "Ages of family members you
want to cover" - and it let "The premium is 12,499" through unchecked,
because a comma defeated the number regex. Wrong about a question, silent
about a price.

Deciding whether a sentence asserts something, and whether a passage supports
it, is a language problem. A regex cannot tell an assertion from a question,
a quotation, or the customer's own words repeated back. A model can - though
not the smallest one available, for reasons measured at the call below.

**What this does NOT do.** It never decides whether a citation is valid. The
model is handed passages numbered [1]..[k] and code checks the number against
what was actually sent - array membership, not judgement. Asking a model "is
[2] one of these four passages" would invite it to hallucinate the one thing
that has to be certain.

**The draft is numbered too.** The model is shown the reply as numbered
sentences and answers with numbers, for the same reason the passages are
numbered: it never has to reproduce our text, so nothing has to be matched
back loosely. Asking a small model to quote sentences exactly produced
verdicts naming sentences that were never written, and a verdict we cannot
locate is a check that did not work.

**Failure is loud, and degrades rather than disappears.** If this is
unavailable the turn proceeds, the trace records that verification did not
run, and the citation-resolution rule still applies. That is weaker, and
saying so is better than either silently failing open or refusing every turn
because a model endpoint was slow.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field

from app.config import settings

log = logging.getLogger("mcb.verify")

SYSTEM = """You check an insurance assistant's draft reply against the source \
passages it was given. You are not writing the reply and you are not \
improving it.

A claim is UNSUPPORTED when the draft states something about the product - \
cover, an exclusion, an amount, a waiting period, a premium, a status or a \
decision - that the passages do not say.

First ask: does this sentence tell the customer something about the product \
that they could later hold us to? If it does not, it is NOT a claim, and \
reporting it deletes a sentence that was never a risk.

These are NOT claims, and must never be reported - even when they are long, \
conditional, or contain product words like "cover" or "waiting period":
- a question to the customer, or a request for their details
- an offer to do something, or to look something up. "If you give me your \
policy ID I can check your waiting period status for you" is an offer, not \
a claim about waiting periods
- repeating back what the customer just said
- a general statement carrying no specific commitment
- a fact listed under ESTABLISHED below, which was confirmed earlier in this \
same conversation

Do report a sentence that asserts a specific fact the passages do not \
support, including one that drops a condition the passages attach to it.

The draft is given to you as numbered sentences. Reply with JSON only, \
naming the numbers of the sentences that are unsupported claims:
{"unsupported": [2, 5]}
An empty list means everything checks out."""


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
    """Take the outermost object. Models wrap JSON in prose and fences.

    Brace to brace rather than a pattern - finding the object is counting
    characters, not recognising a shape.
    """
    t = text or ""
    start, end = t.find("{"), t.rfind("}")
    if start < 0 or end <= start:
        return {}
    try:
        return json.loads(t[start:end + 1])
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

    from app.agents.citations import split
    from app.llm.bedrock import get_llm

    sentences = split(answer)
    if not sentences:
        return Verdict(ran=True)

    numbered = "\n".join(f"{i}. {s.strip()}"
                         for i, s in enumerate(sentences, 1))

    body = [f"PASSAGES\n{_passages(chunks) or '(none were retrieved)'}"]
    if established:
        body.append("ESTABLISHED EARLIER IN THIS CONVERSATION\n"
                    + "\n".join(f"- {e}" for e in established[-12:]))
    body.append(f"DRAFT REPLY, ONE SENTENCE PER LINE\n{numbered}")

    try:
        # Not the small model. Measured over 8 cases x 10 runs: the small
        # model deleted "if you give me your policy ID I can check that for
        # you" as an unsupported claim about waiting periods, 10 times out of
        # 10, while the main model got every case right. Both were stable, so
        # this is capacity rather than prompting - and a gate that decides
        # whether an answer reaches a customer is the wrong place to save a
        # fraction of a cent per turn.
        resp = get_llm(guardrail=False).invoke([
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

    # Numbers back to the sentences they name. The model never has to
    # reproduce our text, so there is nothing to match loosely - the same
    # reason the passages are numbered rather than named.
    picked: list[str] = []
    for x in raw:
        try:
            n = int(str(x).strip().rstrip("."))
        except ValueError:
            continue
        if 1 <= n <= len(sentences):
            s = sentences[n - 1]
            if s not in picked:
                picked.append(s)

    if raw and not picked:
        # It reported something we cannot locate. That is a check that did
        # not work, not an answer that passed.
        return Verdict(ran=False, error="unparseable_verdict")

    return Verdict(ran=True, unsupported=picked)
