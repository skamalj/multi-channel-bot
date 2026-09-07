"""Routing: deterministic first, hysteresis on the prior, a question before
a guess - and a persona that a conversation cannot talk its way into."""
import time

from app.obs.trace import Trace
from app.resolver.resolver import (STEP_UP_TTL_S, resolve, resolve_persona,
                                   refresh_holdings)
from app.resolver.store import ResolverSession


def _sess(user="919899999999"):
    return ResolverSession(user_id=user)


def test_persona_comes_from_the_directory_not_the_text():
    s = _sess("919820000001")          # in PRODUCER_DIRECTORY
    persona, src, q = resolve_persona("919820000001", s, None, Trace())
    assert persona == "agent" and src == "producer_lookup" and q is None

    s2 = _sess()
    persona2, _, _ = resolve_persona(s2.user_id, s2, None, Trace())
    assert persona2 == "customer"


def test_entry_context_cannot_grant_a_persona_the_directory_refuses():
    """RS-9. A deep link is a request, not an answer.

    Without this, `?persona=agent` on a URL is a privilege escalation that
    needs no exploit at all.
    """
    s = _sess()
    t = Trace()
    persona, _, q = resolve_persona(s.user_id, s, "agent", t)
    assert persona == "customer" and q is None
    assert any(e.kind == "guardrail" and e.label == "persona_escalation_refused"
               for e in t.events)


def test_switching_into_agent_needs_a_current_step_up():
    """RS-9: a producer who was being served as a customer cannot slide into
    the producer view without re-authenticating."""
    s = _sess("919820000001")
    s.persona = "customer"                      # was serving their own policy
    t = Trace()
    persona, src, q = resolve_persona(s.user_id, s, None, t)
    assert persona == "customer"                # held, not escalated
    assert src == "step_up_pending" and q is not None
    assert any(e.kind == "gate" and e.label == "step_up_required"
               for e in t.events)

    s.persona_stepup_at = time.time()
    persona2, _, q2 = resolve_persona(s.user_id, s, None, Trace())
    assert persona2 == "agent" and q2 is None
    assert s.persona_switched is True


def test_an_expired_step_up_does_not_count():
    s = _sess("919820000001")
    s.persona = "customer"
    s.persona_stepup_at = time.time() - STEP_UP_TTL_S - 1
    persona, src, q = resolve_persona(s.user_id, s, None, Trace())
    assert persona == "customer" and src == "step_up_pending" and q


def test_explicit_entity_routes_immediately():
    s = _sess()
    spec, q = resolve("my car insurance is due", None, s, "customer", False,
                      Trace())
    assert q is None and spec.lob == "motor"


def test_ambiguous_first_message_asks_rather_than_guesses():
    s = _sess()
    spec, q = resolve("hi", None, s, "customer", False, Trace())
    assert spec is None and q and "?" in q


def test_both_lines_named_in_one_sentence_asks():
    """Two matches is not weak evidence for one of them."""
    s = _sess()
    spec, q = resolve("is my car cover and health cover both due?", None, s,
                      "customer", False, Trace())
    assert spec is None and q


def test_prior_survives_an_ambiguous_turn():
    s = _sess()
    resolve("what is a waiting period on my health plan", None, s,
            "customer", False, Trace())
    assert s.ledger.prior_lob() == "health"
    spec, q = resolve("and what about that", None, s, "customer", False, Trace())
    assert q is None and spec.lob == "health"


def test_threshold_is_higher_when_work_is_in_flight():
    s = _sess()
    resolve("health cover for my family", None, s, "customer", False, Trace())
    assert s.ledger.threshold(True) > s.ledger.threshold(False)


def test_holdings_route_a_first_turn_when_only_one_line_is_held():
    """919820000002 holds a health policy and nothing else."""
    s = _sess("919820000002")
    refresh_holdings(s.user_id, s, Trace())
    assert s.profile.holdings == {"health": True}
    spec, q = resolve("hello", None, s, "customer", False, Trace())
    assert q is None and spec.lob == "health"
    assert s.ledger.last.decided_by == "holdings"


def test_holdings_do_not_decide_when_two_lines_are_held():
    s = _sess("919820000009")           # holds health AND motor
    refresh_holdings(s.user_id, s, Trace())
    spec, q = resolve("hello", None, s, "customer", False, Trace())
    assert spec is None and q


def test_misroute_is_back_annotated_and_becomes_a_labelled_example():
    """RS-7. Turn n routed on the prior; turn n+1 proved it wrong.

    The correction is the golden-set label nobody had to annotate.
    """
    from app.memory.longterm import learning

    s = _sess("919820000002")
    refresh_holdings(s.user_id, s, Trace())
    s.turn = 1
    resolve("hello there", None, s, "customer", False, Trace())   # holdings
    assert s.ledger.last.decided_by == "holdings"

    s.turn = 2
    t = Trace()
    spec, q = resolve("my car needs renewing", None, s, "customer", False, t)
    assert spec.lob == "motor"
    assert s.ledger.events[-2].corrected_to == "motor"
    assert any(e.label == "corrected_previous" for e in t.events)
    assert learning.examples(s.user_id)[0]["actual"] == "motor"


def test_an_explicit_choice_is_not_treated_as_a_misroute():
    """A customer changing subject is two intents, not a wrong route."""
    s = _sess()
    s.turn = 1
    resolve("my car insurance", None, s, "customer", False, Trace())
    s.turn = 2
    resolve("actually about my health policy", None, s, "customer", False,
            Trace())
    assert s.ledger.events[-2].corrected_to is None
