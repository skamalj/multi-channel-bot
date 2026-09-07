"""Bot resolver: persona x lob -> configuration.

Deterministic signals first, the model last, and a question before a guess.
The order in `resolve()` is the requirement RS-3 spells out:

    entry context -> explicit entity -> holdings -> route-ledger prior
                  -> intent model -> ask

**Persona is never inferred from conversation.** It comes from the producer
directory, every turn. Without that line, a long enough conversation becomes
a privilege-escalation path: drift the topic toward agent-shaped questions,
accumulate agent-shaped history, and let the prior grant what an
authorization check would have refused.

And the entry context cannot grant a persona either (RS-9). A deep link
saying `persona=agent` is a *request*; the directory is the *answer*. The
only thing an entry link can do is ask for less than the directory allows.
"""
from __future__ import annotations

import re
import time

from app.agents.registry import REGISTRY, spec_for
from app.coremock import store as core_store
from app.llm.bedrock import classify_lob
from app.obs.trace import Trace
from app.resolver.ledger import RouteEvent
from app.resolver.spec import AgentSpec
from app.resolver.store import ResolverSession

# Deterministic LOB evidence. Explicit statements always outrank the prior.
LOB_PATTERNS: dict[str, re.Pattern] = {
    "motor": re.compile(
        r"\b(car|bike|scooter|vehicle|motor|two[- ]?wheeler|idv|ncb|rto|"
        r"garage|windscreen|bumper|zero[- ]?dep|"
        r"[A-Z]{2}\s?\d{1,2}\s?[A-Z]{1,3}\s?\d{4})\b", re.I),
    "health": re.compile(
        r"\b(health|medical|hospital|mediclaim|cashless|pre[- ]?auth|"
        r"floater|top[- ]?up|ped|pre[- ]?existing|maternity|opd|"
        r"waiting period|sum insured|room rent|co[- ]?pay)\b", re.I),
}

# Stand-in for producer management. Real implementation: a directory lookup
# on the identity, never a model call.
PRODUCER_DIRECTORY: set[str] = {"919820000001", "agent-demo", "rakesh"}

# RS-9: how long a step-up authentication is good for.
STEP_UP_TTL_S = 15 * 60


def directory_persona(user_id: str) -> str:
    return "agent" if user_id in PRODUCER_DIRECTORY else "customer"


def step_up_current(session: ResolverSession) -> bool:
    at = session.persona_stepup_at
    return bool(at and (time.time() - at) < STEP_UP_TTL_S)


def resolve_persona(user_id: str, session: ResolverSession,
                    entry_persona: str | None,
                    trace: Trace) -> tuple[str, str, str | None]:
    """Returns (persona, source, question).

    A question means the turn stops here: a persona switch into `agent`
    needs a step-up authentication first, and until it happens the caller is
    served as the persona they already had.
    """
    allowed = directory_persona(user_id)

    requested = entry_persona
    if requested and requested not in {s.persona for s in REGISTRY.values()}:
        requested = None

    # An entry link may ask for LESS than the directory allows, never more.
    if requested == "agent" and allowed != "agent":
        trace.add("guardrail", "persona_escalation_refused",
                  requested=requested, directory=allowed,
                  detail="entry context cannot grant a persona the producer "
                         "directory does not")
        requested = None

    persona = requested or allowed

    # RS-9: switching INTO agent mid-conversation is a privilege change. It
    # needs a fresh authentication, and it carries no journey state - the
    # orchestrator starts the agent thread clean.
    previous = session.persona
    if previous and persona != previous:
        if persona == "agent" and not step_up_current(session):
            trace.add("gate", "step_up_required", frm=previous, to=persona,
                      detail="a persona switch into agent requires step-up "
                             "authentication")
            return previous, "step_up_pending", (
                "Switching to the producer view needs a quick identity check. "
                "Please confirm your identity to continue, and I will keep "
                "your customer conversation exactly where it is.")
        trace.add("resolve", "persona_switch", frm=previous, to=persona,
                  stepped_up=step_up_current(session),
                  detail="no journey state carries across a persona switch")
        session.persona_switched = True

    trace.add("resolve", "persona", value=persona,
              source="entry" if requested else "producer_lookup",
              directory=allowed)
    return persona, "entry" if requested else "producer_lookup", None


def _lob_evidence(text: str | None) -> tuple[str | None, float, str]:
    if not text:
        return None, 0.0, ""
    hits = {lob: p.search(text) for lob, p in LOB_PATTERNS.items()}
    matched = {lob: m for lob, m in hits.items() if m}
    if len(matched) == 1:
        lob, m = next(iter(matched.items()))
        return lob, 0.95, m.group(0)
    if len(matched) > 1:
        # Both lines named in one sentence is not weak evidence for one of
        # them, it is evidence that a question is needed.
        return None, 0.4, ",".join(sorted(matched))
    return None, 0.0, ""


def refresh_holdings(user_id: str, session: ResolverSession,
                     trace: Trace) -> None:
    """Holdings come from the policy directory, not from the conversation.

    They are facts ABOUT a line of business - which is precisely the class of
    fact the shared profile is allowed to hold.
    """
    held = core_store.holdings_for(user_id)
    if held and held != session.profile.holdings:
        session.profile.holdings = held
        trace.add("resolve", "holdings", value=held, source="policy_directory")


def resolve(event_text: str | None, entry_lob: str | None,
            session: ResolverSession, persona: str,
            work_in_flight: bool,
            trace: Trace) -> tuple[AgentSpec | None, str | None]:
    """Returns (spec, question_to_ask). Exactly one of them is None."""

    prior = session.ledger.prior_lob()
    threshold = session.ledger.threshold(work_in_flight)
    options = sorted({s.lob for s in REGISTRY.values() if s.persona == persona})

    # 1. Entry context - a deep link, QR or embed. Deterministic, bypasses all.
    if entry_lob:
        if entry_lob not in options:
            trace.add("guardrail", "entry_lob_not_configured", value=entry_lob,
                      available=options)
        else:
            return _commit(session, persona, entry_lob, "entry", 1.0,
                           "entry link", prior, event_text, trace), None

    # 2. Explicit statement or hard entity. The user outranks the prior.
    lob, conf, evidence = _lob_evidence(event_text)
    if lob and conf >= 0.9 and lob in options:
        return _commit(session, persona, lob, "explicit", conf, evidence,
                       prior, event_text, trace), None

    # 3. Holdings: one line of business held, and nothing contradicting it.
    held = [k for k, v in session.profile.holdings.items()
            if v and k in options]
    if not lob and len(held) == 1 and prior is None:
        return _commit(session, persona, held[0], "holdings", 0.8,
                       "single holding on the policy directory", prior,
                       event_text, trace), None

    # 4. The prior. A strong history is real evidence; do not overturn it on
    #    an ambiguous sentence.
    if prior and (lob is None or conf < threshold):
        trace.add("resolve", "lob", value=prior, source="history",
                  threshold=threshold, challenger=lob, challenger_conf=conf)
        session.active_lob = prior
        session.ledger.append(
            RouteEvent(turn=session.turn, persona=persona, lob=prior,
                       decided_by="history", confidence=1 - conf,
                       evidence="prior held against a weaker challenger",
                       text=(event_text or "")[:300]))
        return spec_for(persona, prior), None

    # 5. The intent model - the LAST resort before asking, and it fails
    #    closed. Everything above this line is deterministic and auditable;
    #    this is the only step where a model has an opinion about routing,
    #    and its opinion still has to clear the threshold.
    if lob is None:
        mlob, mconf, mevidence = classify_lob(event_text or "", options)
        trace.add("resolve", "intent_model", value=mlob, confidence=mconf,
                  evidence=mevidence, threshold=threshold)
        if mlob and mconf >= threshold:
            return _commit(session, persona, mlob, "intent_model", mconf,
                           mevidence or "intent model", prior, event_text,
                           trace), None

    # 6. Nothing decided. One question beats a wrong route - and the answer is
    #    the highest-quality label there is.
    trace.add("resolve", "lob", value=None, source="asked_user",
              threshold=threshold, ambiguous_evidence=evidence)
    pretty = " or ".join(options)
    return None, f"Happy to help. Is this about your {pretty} cover?"


def _commit(session: ResolverSession, persona: str, lob: str, decided_by: str,
            conf: float, evidence: str, prior: str | None,
            text: str | None, trace: Trace) -> AgentSpec:
    # RS-7: if the previous turn routed on weak evidence and this turn proves
    # it wrong, label the previous decision before appending this one. That
    # turns every mis-route into a training example nobody had to annotate.
    if prior and prior != lob and decided_by in ("explicit", "entry",
                                                 "asked_user"):
        corrected = session.ledger.back_annotate(lob, session.turn)
        if corrected:
            trace.add("resolve", "corrected_previous", frm=corrected.lob,
                      to=lob, was_decided_by=corrected.decided_by,
                      turn=corrected.turn)
            try:
                from app.memory.longterm import learning

                learning.record_correction(
                    session.user_id, corrected.text, corrected.lob, lob,
                    corrected.decided_by, corrected.confidence)
            except Exception:                                # noqa: BLE001
                pass       # a learning-store failure must not fail the turn

    if prior and prior != lob:
        # A switch. The old bot session is untouched and still exactly where
        # it was - suspension is free when the sessions are keyed apart.
        trace.add("resolve", "switch", frm=prior, to=lob, reason=decided_by)

    session.ledger.append(
        RouteEvent(turn=session.turn, persona=persona, lob=lob,
                   decided_by=decided_by, confidence=conf, evidence=evidence,
                   text=(text or "")[:300],
                   switched_from=prior if prior != lob else None)
    )
    session.persona = persona
    session.active_lob = lob
    trace.add("resolve", "lob", value=lob, source=decided_by, confidence=conf,
              evidence=evidence)
    return spec_for(persona, lob)
