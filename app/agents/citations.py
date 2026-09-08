"""Citation enforcement (KB-4) and the refusal path (KB-5).

The rule: when a turn retrieved, every claim in the answer must be supported
by a passage that was actually retrieved, and every citation must point at
one of those passages.

**Two jobs, two mechanisms, and the split is the whole design.**

*Is this sentence a claim, and do the passages support it?* is a language
question, and it lives in `app/agents/verify.py` on a small model. The
previous version of this file tried to answer it with word lists, and failed
in both directions in a single turn: it refused the bot's own question -
"Ages of family members you want to **cover**" - because the word "cover"
appeared in it, while letting "The premium is 12,499" through completely
unchecked, because a comma defeated the number regex. Wrong about a question,
silent about a price. A regex cannot tell an assertion from a question, and
no amount of patching will teach it to.

*Does `[2]` name a passage we actually sent?* is not a language question at
all. It is array membership, and it stays here in code. Asking a model that
would invite it to hallucinate the one fact that has to be certain.

**Citations are NUMBERS now.** The retrieval tool hands the model passages
labelled `[1]`..`[k]`, not `PHS-POLICY_WORDING-V2#1`. That change removed a
whole class of failure: the model used to paraphrase the structured id -
`PHS-POLICYWORDING-V21`, underscore and hash gone - and a correct, properly
sourced answer was refused because a string comparison failed. There is no
fuzzy way to write `[2]`.

What remains here:

1. A citation naming a passage that was never sent is stripped. A fake
   citation is worse than none, because it looks like provenance.
2. Sentences the verifier reports as unsupported are dropped.
3. If everything material was dropped, or retrieval ran and returned nothing,
   the turn refuses and offers a colleague. An empty answer is a correct
   answer.
"""
from __future__ import annotations

import re

# A bracket may hold more than one reference - "[1, 3]" is how a model
# naturally cites two passages for one sentence.
BRACKET = re.compile(r"\[([^\[\]]{1,40})\]")
REF = re.compile(r"\b(\d{1,2})\b")

REFUSAL = (
    "I could not find anything in our documented sources that answers that, "
    "so I would rather not guess. I can put you through to a colleague who "
    "can check it properly - shall I do that?")

# The same refusal aimed at an ACTION is misleading. A customer who asked to
# issue a policy and is told "I do not have an approved source" hears a
# knowledge gap, when what happened is that nothing was done.
REFUSAL_ACTION = (
    "I have not done that - I could not complete it from what I have here, "
    "and I would rather tell you than let you think it went through. Shall I "
    "put you through to a colleague who can finish it?")

_SENTENCE = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\[*#\-])")


# ---------------------------------------------------------------------------
# citation references
# ---------------------------------------------------------------------------
def refs_in(text: str) -> list[tuple[str, list[int]]]:
    """Every bracket in `text`, with the passage numbers inside it."""
    out: list[tuple[str, list[int]]] = []
    for m in BRACKET.finditer(text or ""):
        inner = m.group(1)
        nums = [int(n) for n in REF.findall(inner)]
        if nums:
            out.append((m.group(0), nums))
    return out


def strip_unknown_refs(text: str, valid: set[int]) -> tuple[str, list[int]]:
    """Remove brackets naming passages that were never sent.

    Returns the cleaned text and the invented numbers, which belong in the
    trace: a model citing `[7]` when four passages were given has fabricated
    provenance, and that is worth seeing even though the sentence may survive
    on the verifier's judgement.
    """
    invented: list[int] = []
    cleaned = text or ""
    for bracket, nums in refs_in(cleaned):
        unknown = [n for n in nums if n not in valid]
        if not unknown:
            continue
        invented.extend(unknown)
        keep = [n for n in nums if n in valid]
        replacement = ("[" + ", ".join(str(n) for n in keep) + "]") if keep else ""
        cleaned = cleaned.replace(bracket, replacement)
    return re.sub(r"[ \t]{2,}", " ", cleaned).strip(), sorted(set(invented))


def cited_chunk_ids(text: str, chunks: list[dict]) -> list[str]:
    """The real chunk ids behind the numbers the model cited.

    The audit record names documents, not positions - `[2]` means nothing six
    months from now, and the whole point of a citation is that somebody can
    go and read the thing.
    """
    by_ref = {c["ref"]: c.get("chunk_id", "") for c in chunks if "ref" in c}
    out: list[str] = []
    for _, nums in refs_in(text):
        for n in nums:
            cid = by_ref.get(n)
            if cid and cid not in out:
                out.append(cid)
    return out


def split(answer: str) -> list[str]:
    """Sentence split that leaves list items and headings alone."""
    out: list[str] = []
    for line in (answer or "").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if re.match(r"^([-*•]|\d+[.)]|#)\s*", stripped):
            out.append(line)
        else:
            out.extend(_SENTENCE.split(line))
    return [o for o in out if o.strip()]


def _looks_structured(lines: list[str]) -> bool:
    return any(l.strip().startswith(("#", "**", "-", "*", "•")) or
               re.match(r"^\s*\d+[.)]\s", l) for l in lines)


def _join(kept: list[str]) -> str:
    """Keep the model's shape when it wrote one - headings and bullets read
    as a list, not as a paragraph with asterisks in the middle of it."""
    return ("\n".join(kept).strip() if _looks_structured(kept)
            else " ".join(kept).strip())


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[*_`#>]+", "", s or "")).strip().lower()


# ---------------------------------------------------------------------------
# enforcement
# ---------------------------------------------------------------------------
def enforce(answer: str, chunks: list[dict],
            other_tool_evidence: bool = False,
            established: list[str] | None = None,
            retrieval_ran: bool = True,
            action_attempted: bool = False) -> tuple[str, dict]:
    """Returns (answer, report).

    `chunks` is what retrieval actually returned, each carrying a `ref`.
    `other_tool_evidence` is True when a core tool ran - a premium, a policy
    record, a gate decision. Those sentences are grounded in a tool result
    rather than a document, and the tool call is itself in the trace.
    `established` is what this conversation already confirmed with a source,
    so a follow-up may restate it.
    `action_attempted` picks the refusal wording, and it means what it says:
    a state-changing tool ran. Choosing it by whether retrieval happened -
    as this once did - told a customer asking a question "I have not done
    that", describing a transaction they never started.
    """
    from app.agents import verify

    report: dict = {"cited": [], "dropped": [], "invented_refs": [],
                    "verifier": "not_run", "refused": False}

    valid = {c["ref"] for c in chunks if "ref" in c}

    # 1. A citation naming a passage we never sent is provenance theatre.
    answer, invented = strip_unknown_refs(answer, valid)
    report["invented_refs"] = invented

    # Nothing retrieved and nothing else to stand on: there is no answer to
    # give, whatever the model wrote.
    if retrieval_ran and not chunks and not other_tool_evidence:
        report["refused"] = True
        return (REFUSAL_ACTION if action_attempted else REFUSAL), report

    # 2. Does the source material support what was written?
    verdict = verify.check(answer, chunks, established=established)
    report["verifier"] = ("ran" if verdict.ran
                          else (verdict.error or "not_configured"))

    if not verdict.ran:
        # Degraded, and the trace says so. The reference check above still
        # applied, so a fabricated citation was still removed.
        report["cited"] = cited_chunk_ids(answer, chunks)
        return answer, report

    unsupported = {_norm(u) for u in verdict.unsupported}
    if not unsupported:
        report["cited"] = cited_chunk_ids(answer, chunks)
        return answer, report

    kept: list[str] = []
    dropped: list[str] = []
    for sentence in split(answer):
        if _norm(sentence) in unsupported:
            dropped.append(sentence.strip()[:160])
        else:
            kept.append(sentence)

    # The verifier quotes sentences; if none matched, its quoting was loose
    # rather than the answer being clean, so treat the whole answer as
    # unsupported rather than silently passing it.
    if not dropped:
        dropped = [u[:160] for u in verdict.unsupported]
        kept = []

    report["dropped"] = dropped

    if not kept:
        report["refused"] = True
        return (REFUSAL_ACTION if action_attempted else REFUSAL), report

    text = _join(kept)
    report["cited"] = cited_chunk_ids(text, chunks)
    return text, report
