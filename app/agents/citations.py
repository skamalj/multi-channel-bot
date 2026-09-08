"""Citation enforcement (KB-4) and the refusal path (KB-5).

The rule: when a turn retrieved, every claim in the answer must be supported
by a passage that was actually retrieved, and every citation must point at
one of those passages.

**Two jobs, two mechanisms, and the split is the whole design.**

*Is this sentence a claim, and do the passages support it?* is a language
question, and it lives in `app/agents/verify.py` on a model. The version of
this file before that failed in both directions in a single turn: it refused
the bot's own question - "Ages of family members you want to **cover**" -
because the word "cover" appeared in it, while letting "The premium is
12,499" through completely unchecked, because a comma defeated the number
regex. Wrong about a question, silent about a price.

*Does `[2]` name a passage we actually sent?* is not a language question at
all. It is array membership, and it stays here in code. Asking a model that
would invite it to hallucinate the one fact that has to be certain.

**Citations are NUMBERS.** The retrieval tool hands the model passages
labelled `[1]`..`[k]`, not `PHS-POLICY_WORDING-V2#1`. That removed a whole
class of failure: the model used to paraphrase the structured id -
`PHS-POLICYWORDING-V21`, underscore and hash gone - and a correct, properly
sourced answer was refused because a string comparison failed. There is no
fuzzy way to write `[2]`.

What remains here:

1. A citation naming a passage that was never sent is stripped. A fake
   citation is worse than none, because it looks like provenance.
2. Units the verifier reports as unsupported are dropped.
3. If everything material was dropped, or retrieval ran and returned nothing,
   the turn refuses and offers a colleague. An empty answer is a correct
   answer.

**No regular expressions.** Not a style preference. Every pattern that has
ever lived in this file ended up deciding something about language and
getting it wrong, and each time a customer's answer was worse for it. What is
left is scanning a bracket format we chose ourselves, and splitting on line
breaks. Both are counting characters. Everything that needs a sentence to be
read is in `verify.py`, on a model.
"""
from __future__ import annotations

# A bracket may hold more than one reference - "[1, 3]" is how a model
# naturally cites two passages for one sentence. Anything longer than this is
# prose in brackets, not a citation.
MAX_BRACKET = 40

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


# ---------------------------------------------------------------------------
# citation references
# ---------------------------------------------------------------------------
def refs_in(text: str) -> list[tuple[str, list[int]]]:
    """Every bracket in `text`, with the passage numbers inside it.

    Scanned, not matched. This is reading back a format we chose and told the
    model to use - find a bracket, read the numbers in it - and decides
    nothing about what the sentence means.
    """
    out: list[tuple[str, list[int]]] = []
    t = text or ""
    i = 0
    while True:
        open_at = t.find("[", i)
        if open_at < 0:
            break
        close_at = t.find("]", open_at + 1)
        if close_at < 0:
            break
        inner = t[open_at + 1:close_at]
        i = close_at + 1
        if len(inner) > MAX_BRACKET:
            continue
        nums = _numbers_in(inner)
        if nums:
            out.append((t[open_at:close_at + 1], nums))
    return out


def _numbers_in(inner: str) -> list[int]:
    """The passage numbers inside one bracket. `[1, 3]` is two of them."""
    nums: list[int] = []
    for token in inner.replace(",", " ").replace(";", " ").split():
        if token.isdigit() and len(token) <= 2:
            nums.append(int(token))
    return nums


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
    return _tidy(cleaned), sorted(set(invented))


def _tidy(text: str) -> str:
    """Close the gap a removed bracket leaves, keeping any indentation."""
    out = []
    for line in (text or "").splitlines():
        lead = line[:len(line) - len(line.lstrip())]
        out.append((lead + " ".join(line.split())).rstrip())
    return "\n".join(out).strip()


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
    """The units the verifier is shown, and the units that can be dropped.

    Lines, not sentences. Where a sentence ends is a language question, and
    the pattern that used to answer it here - break after . ! or ? when a
    capital follows - splits "Rs 24,780. Dr. Rao confirmed" in the wrong
    place. That mattered more than it looks: the verifier replies with the
    NUMBER of a unit, so a unit split in the wrong place deletes the wrong
    text from a customer's answer.

    A line is coarser, and coarse in the safe direction - a dense paragraph
    is kept or dropped whole, and half a sentence can never be removed with
    the other half left standing. These answers are written as headings and
    bullets anyway, so in practice a line is already close to one claim.
    """
    return [line for line in (answer or "").splitlines() if line.strip()]


def _looks_structured(lines: list[str]) -> bool:
    """Did the model write a list, or a paragraph? Layout only."""
    for line in lines:
        stripped = line.strip()
        if stripped.startswith(("#", "**", "-", "*", "•")):
            return True
        head = stripped.split(".", 1)[0].split(")", 1)[0]
        if head.isdigit() and len(head) <= 2 and len(stripped) > len(head):
            return True
    return False


def _join(kept: list[str]) -> str:
    """Keep the model's shape when it wrote one - headings and bullets read
    as a list, not as a paragraph with asterisks in the middle of it."""
    return ("\n".join(kept).strip() if _looks_structured(kept)
            else " ".join(kept).strip())


def _norm(s: str) -> str:
    """Compare on words, ignoring markdown emphasis and spacing."""
    text = s or ""
    for ch in "*_`#>":
        text = text.replace(ch, "")
    return " ".join(text.split()).strip().lower()


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

    # Retrieval ran and found nothing, and no tool result stands behind the
    # turn either. That USED to refuse here, before reading what the model
    # wrote - and a vague follow-up is exactly when it bites. "What are other
    # benefits of this" scores below the retrieval floor, nothing comes back,
    # and the model quite correctly writes "which plan did you mean?" - which
    # was then thrown away and replaced with "I could not find anything in
    # our documented sources". A question refused for lacking a citation is
    # the same fault as refusing "ages of family members you want to cover".
    #
    # The verifier decides instead. Measured with NO passages at all, 6 runs
    # each: a clarifying question and an offer to look something up came back
    # clean 6/6; an invented benefit, an invented premium, and a question
    # with a claim attached were all flagged 6/6. Given nothing to cite, it
    # flags everything that needed citing - which is the whole rule.
    nothing_to_stand_on = (retrieval_ran and not chunks
                           and not other_tool_evidence)

    # 2. Does the source material support what was written?
    verdict = verify.check(answer, chunks, established=established)
    report["verifier"] = ("ran" if verdict.ran
                          else (verdict.error or "not_configured"))

    if not verdict.ran:
        # Degraded, and the trace says so. Without a verifier nothing here
        # can tell a question from a claim, so the blunt rule comes back:
        # when there was nothing to stand on, refuse rather than pass an
        # unchecked answer.
        if nothing_to_stand_on:
            report["refused"] = True
            return (REFUSAL_ACTION if action_attempted else REFUSAL), report
        # The reference check above still applied, so a fabricated citation
        # was still removed.
        report["cited"] = cited_chunk_ids(answer, chunks)
        return answer, report

    unsupported = {_norm(u) for u in verdict.unsupported}
    if not unsupported:
        report["cited"] = cited_chunk_ids(answer, chunks)
        return answer, report

    kept: list[str] = []
    dropped: list[str] = []
    for unit in split(answer):
        if _norm(unit) in unsupported:
            dropped.append(unit.strip()[:160])
        else:
            kept.append(unit)

    # The verifier names units from this same split, so a verdict that
    # matches nothing means the check did not do its job. Treat the whole
    # answer as unsupported rather than silently passing it.
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
