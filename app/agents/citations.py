"""Citation enforcement (KB-4) and the refusal path (KB-5).

The rule: when a turn retrieved, every **material claim** in the answer must
be traceable to a chunk that was actually retrieved. Uncited claims are
dropped before the answer is composed, not flagged afterwards.

What counts as a material claim is the whole design of this file. The naive
reading - "every sentence needs a citation" - deletes headings, hedges and
offers of help, and produces answers that are safe and unreadable. The claims
that hurt somebody in this domain are **specific**: a figure, a percentage, a
money amount, a date, a promise that something is covered, excluded, free or
unlimited. Those need a source. "It depends on when your policy started" does
not.

Order of decisions per sentence:

1. Strip markers naming chunks we did not retrieve. A fake citation is worse
   than none, because it looks like provenance.
2. A marker for a chunk we really retrieved -> keep.
3. A heading, a hedge, a question, an offer of help -> keep. It makes no
   material claim, so there is nothing to cite.
4. A material claim with no marker, but substantially the retrieved text ->
   keep and ATTACH the citation. The sentence was grounded; the formatting
   was not, and deleting a correct sentence is the worse error.
5. A material claim with no support -> drop, and record it in the trace.

If retrieval ran and returned nothing, or every material claim was dropped,
the turn refuses and offers a human. An empty answer is a correct answer.
"""
from __future__ import annotations

import re

# A citation bracket may hold more than one id - "[A#1, B#2]" is how a model
# naturally cites two sources for one sentence, and matching only single-id
# brackets silently discarded correctly grounded answers.
BRACKET = re.compile(r"\[([^\[\]]{3,})\]")
ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9\-.#_]{3,}")


def _looks_like_id(part: str) -> bool:
    """A chunk id, not an English word.

    "[above]" is an aside and must survive; treating any bracketed word as a
    failed citation deletes ordinary prose. Every id in this corpus carries a
    `#section` suffix, and a bare document id is upper-case and hyphenated.
    """
    if not ID.fullmatch(part):
        return False
    if "#" in part:
        return True
    return "-" in part and part == part.upper()


_VERSION_SEG = re.compile(r"-V\d+(?=#|$)", re.I)


def _resolve_id(cited: str, valid: set[str]) -> str | None:
    """A cited id that is a near-miss for one we really retrieved.

    Only PHS has dated wording versions, so its chunks are
    `PHS-POLICY_WORDING-V2#1` while every other product's are
    `PHST-POLICY_WORDING#1`. A model reading both generalises and writes the
    versionless form. That is a typo pointing at a chunk that WAS retrieved,
    not an invented source, and treating it as a hallucination throws away a
    correctly grounded answer.

    Resolved only when it is UNAMBIGUOUS. If both V1 and V2 were retrieved,
    the model has not said which wording it means, and that is exactly the
    distinction KB-6 exists to keep.
    """
    target = _VERSION_SEG.sub("", cited)
    matches = {v for v in valid if _VERSION_SEG.sub("", v) == target}
    return matches.pop() if len(matches) == 1 else None


def _citations_in(text: str) -> list[tuple[str, list[str]]]:
    """(whole bracket, ids inside it) for every bracket that IS a citation."""
    out = []
    for m in BRACKET.finditer(text):
        parts = [p.strip() for p in re.split(r"[,;]", m.group(1))]
        ids = [p for p in parts if _looks_like_id(p)]
        if ids:
            out.append((m.group(0), ids))
    return out


_SENTENCE = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\[*#\-])")
_WORD = re.compile(r"[a-z0-9]+")
_MD = re.compile(r"[*_`#>]+")

REPAIR_OVERLAP = 0.5

# A figure that is not touching a letter. "Q-9D57E12C" is an identifier, not
# the numbers 9, 57 and 12, and treating it as three figures would reject a
# quote id for not appearing in its own quote.
_NUMBER = re.compile(r"(?<![A-Za-z0-9])\d[\d,]*(?:\.\d+)?(?![A-Za-z0-9])")

# A claim that can hurt somebody is a specific one - and specifically, it is
# a QUANTITY or a promise, not any character that happens to be a digit.
#
# "any figure" was the first version of this and it was wrong in a way that
# only showed up in a transactional conversation: "Application ID: A-47C234A2"
# and "1. KYC verification" both contain digits, so a bot walking a customer
# through issuance had every sentence treated as an unsourced claim and
# refused its way out of its own journey. An identifier is a reference, not an
# assertion about cover.
#
# Two kinds, and the difference decides what can rescue them. A QUANTITY is
# checkable against what a tool returned. A PROMISE is not - "unlimited
# overseas cover" has no figure to verify, so nothing but a cited source can
# support it.
_QUANTITY = [
    re.compile(r"(?:\brs\.?\s*|₹)\s*[\d,]+", re.I),          # money
    re.compile(r"\d+\s*(?:%|per ?cent)", re.I),               # percentage
    re.compile(r"\b\d+\s*(?:hour|day|week|month|year)s?\b", re.I),
    re.compile(r"(?<![A-Za-z0-9])\d{4,}(?![A-Za-z0-9])"),     # amounts, years
    re.compile(r"\b(lakh|crore)\b", re.I),
]
_PROMISE = [
    re.compile(r"\b(cover(?:ed|s)?|exclud(?:ed|es)|payable|admissible|"
               r"reimburse\w*|refund\w*|includes?|entitled|eligible|"
               r"waiting period|no claim bonus)\b", re.I),
    re.compile(r"\b(free|complimentary|unlimited|guaranteed|bonus|cashback|"
               r"discount|waived|lifetime)\b", re.I),
    # A DECISION belongs to the core, and its vocabulary is closed. A model
    # that writes "Underwriting Decision: CLEARED" without calling
    # underwriting has told the customer the most consequential thing in the
    # journey, and it is not true. The gate caught that two turns later - by
    # which point they had been told.
    re.compile(r"\b(cleared|approved|issued|sanctioned|settled|declined|"
               r"repudiated|rejected|in force)\b", re.I),
]
_MATERIAL = _QUANTITY + _PROMISE

# Deliberately avoids the word "approved". It is in the decision vocabulary
# below - "approved" is what a claim or an underwriting file is - so a
# refusal phrased with it trips the very check it came from, and any caller
# re-inspecting the final reply sees an unsupported decision word.
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


# A list marker is formatting. Leaving it in makes "1. Your registration
# number" a sentence containing a figure, so a bot asking two clarifying
# questions in a numbered list looked like it was making two unsourced
# claims - and the whole turn refused.
_LIST_MARKER = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+")


def _plain(s: str) -> str:
    """Formatting off, so the tests below see the sentence, not its markup."""
    return _LIST_MARKER.sub("", _MD.sub("", s)).strip()


def _tokens(text: str) -> set[str]:
    return {w for w in _WORD.findall(text.lower()) if len(w) > 3}


def _numbers(text: str) -> set[str]:
    return {n.replace(",", "").rstrip(".") for n in _NUMBER.findall(text or "")}


def _figures_supported(sentence: str, facts: str) -> bool:
    """Every figure in the sentence appears in the tool results behind it.

    This is the check that catches the expensive class of drift: a model
    restating a premium, an IDV, a date or a registration year from its own
    earlier prose rather than from what the core just returned. A quote whose
    total is off by a rupee is a mis-sold policy, and no amount of fluent
    formatting around it makes it less so.
    """
    quoted = _numbers(sentence)
    if not quoted:
        return True
    known = _numbers(facts)
    known_floats = []
    for k in known:
        try:
            known_floats.append(float(k))
        except ValueError:                                   # pragma: no cover
            pass
    for q in quoted:
        if q in known:
            continue
        try:
            qf = float(q)
        except ValueError:
            return False
        # A sensible rounding of a real figure is still that figure.
        if not any(abs(qf - k) <= 0.51 for k in known_floats):
            return False
    return True


def _is_structural(s: str) -> bool:
    """A heading or a lead-in. It introduces claims; it does not make one.

    A clause ending in a colon is structural however long it is. "IRDAI has
    set the maximum waiting period for pre-existing diseases at:" mentions a
    waiting period and cites nothing, so a naive materiality test drops it -
    and orphans the three cited bullets it was introducing, which is a worse
    answer than either keeping or dropping the whole passage.
    """
    raw = s.strip()
    if raw.startswith("#"):
        return True
    if raw.startswith("**") and raw.endswith("**"):
        return True
    return _plain(raw).endswith(":")


def _is_material(s: str) -> bool:
    plain = _plain(s)
    return any(p.search(plain) for p in _MATERIAL)


def _is_promise(s: str) -> bool:
    """A claim about cover, with no figure in it to check."""
    plain = _plain(s)
    return any(p.search(plain) for p in _PROMISE)


def _is_question(s: str) -> bool:
    plain = _plain(s)
    return not plain or plain.endswith("?")


def _needs_source(s: str) -> bool:
    """A sentence that must be traceable to something.

    Note what is NOT here: a length shortcut. "The waiting period is two
    weeks." is six words and is exactly the sentence this file exists to
    stop, so brevity cannot be a reason to trust a claim.
    """
    return _is_material(s) and not _is_structural(s) and not _is_question(s)


def enforce(answer: str, chunks: list[dict],
            other_tool_evidence: bool = False,
            tool_facts: str = "",
            retrieval_ran: bool = True) -> tuple[str, dict]:
    """Returns (answer, report). `chunks` is what retrieval actually returned.

    `other_tool_evidence` is True when the turn also called a core tool - a
    premium, a policy record, a gate decision. Those sentences are grounded
    in a tool result rather than a document, and the tool call is itself in
    the trace, so they are not dropped for want of a chunk id.

    `tool_facts` is what those tools actually returned, and it is checked
    rather than assumed. Treating "a core tool ran" as a blanket exemption
    was a real hole: it let a quote summary say "Maruti Swift VXI" over a
    lookup that returned a Baleno, because one true sentence about the
    premium licensed every other sentence in the same answer. It carries the
    facts established EARLIER in the same journey too - an application id
    quoted back three turns after the core returned it is a reference, not a
    new claim.

    `retrieval_ran` only picks the wording of a refusal: a customer who asked
    to issue a policy and is told "I do not have an approved source" hears a
    knowledge gap rather than "nothing was done".
    """
    valid = {c["chunk_id"] for c in chunks}
    report: dict = {"cited": [], "dropped": [], "repaired": [],
                    "hallucinated": [], "refused": False}

    if not chunks:
        # No documents behind this answer. That covers two cases and they are
        # NOT the same: a core tool answered (a premium, a policy record), or
        # nothing did.
        if other_tool_evidence:
            # A core tool answered. Its figures are the truth, so every
            # figure in the prose has to be one of them.
            lines = _split(answer)
            kept = [ln for ln in lines
                    if not _needs_source(ln)
                    or _figures_supported(ln, tool_facts)]
            report["dropped"] = [_plain(ln)[:160] for ln in lines
                                 if ln not in kept]
            return _join(kept) or answer, report
        # Nothing did. Only text that makes no material claim may stand - a
        # greeting or a clarifying question is a fine answer with no sources;
        # "health insurance does not usually cover that" is not, however
        # reasonable it sounds, because nothing here checked it.
        # A sentence whose only "material" content is a figure this journey
        # already established - an application id, the quote total the core
        # returned two turns ago - is a reference, not a new claim. A PROMISE
        # has no figure to check and can only be supported by a source.
        unsupported = [_plain(s) for s in _split(answer)
                       if _needs_source(s)
                       and (_is_promise(s)
                            or not _figures_supported(s, tool_facts))]
        if not unsupported:
            return answer, report
        report["dropped"] = [u[:160] for u in unsupported]
        report["refused"] = True
        return (REFUSAL if retrieval_ran else REFUSAL_ACTION), report

    kept: list[str] = []
    material_kept = 0
    for sentence in _split(answer):
        raw = sentence.strip()
        if not raw:
            continue

        found: list[str] = []
        for bracket, ids in _citations_in(raw):
            good, bad, fixed = [], [], False
            for i in ids:
                if i in valid:
                    good.append(i)
                    continue
                resolved = _resolve_id(i, valid)
                if resolved:
                    good.append(resolved)
                    fixed = True
                else:
                    bad.append(i)
            report["hallucinated"].extend(bad)
            found.extend(good)
            if bad or fixed:
                # Keep the real ids, drop the invented ones. A fake marker is
                # worse than none because it looks like provenance.
                raw = raw.replace(bracket,
                                  f"[{', '.join(good)}]" if good else "")
        raw = re.sub(r"\s{2,}", " ", raw).strip()

        if found:
            report["cited"].extend(found)
            kept.append(raw)
            material_kept += 1
            continue

        if not _needs_source(raw):
            kept.append(raw)
            continue

        if other_tool_evidence:
            if _figures_supported(raw, tool_facts):
                kept.append(raw)
                material_kept += 1
            else:
                report["dropped"].append(_plain(raw)[:160])
            continue

        support = _best_support(raw, chunks)
        if support:
            chunk_id, overlap = support
            report["repaired"].append({"chunk_id": chunk_id,
                                       "overlap": round(overlap, 2)})
            report["cited"].append(chunk_id)
            kept.append(f"{raw.rstrip('.')} [{chunk_id}].")
            material_kept += 1
            continue

        report["dropped"].append(_plain(raw)[:160])

    if not material_kept:
        report["refused"] = True
        return REFUSAL, report
    return _join(kept), report


def _join(kept: list[str]) -> str:
    """Keep the model's shape when it wrote one - headings and bullets read
    as a list, not as a paragraph with asterisks in the middle of it."""
    return ("\n".join(kept).strip() if _looks_structured(kept)
            else " ".join(kept).strip())


def _looks_structured(lines: list[str]) -> bool:
    """Keep the model's shape when it wrote one - headings and bullets read
    as a list, not as a paragraph with asterisks in it."""
    return any(l.strip().startswith(("#", "**", "-", "*", "•")) or
               re.match(r"^\s*\d+[.)]\s", l) for l in lines)


def _split(answer: str) -> list[str]:
    """Sentence split that leaves list items and headings alone."""
    out: list[str] = []
    for line in (answer or "").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if re.match(r"^([-*•]|\d+[.)]|#)\s*", stripped) or _is_structural(line):
            out.append(line)
        else:
            out.extend(_SENTENCE.split(line))
    return [o for o in out if o.strip()]


def _best_support(sentence: str, chunks: list[dict]) -> tuple[str, float] | None:
    """The chunk this sentence was plainly taken from, if there is one."""
    st = _tokens(_plain(sentence))
    if len(st) < 4:
        return None
    best, best_overlap = None, 0.0
    for c in chunks:
        ct = _tokens(c.get("text", ""))
        if not ct:
            continue
        overlap = len(st & ct) / len(st)
        if overlap > best_overlap:
            best, best_overlap = c["chunk_id"], overlap
    return (best, best_overlap) if best and best_overlap >= REPAIR_OVERLAP \
        else None
