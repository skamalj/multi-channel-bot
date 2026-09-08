"""One buying journey, replayed against the DEPLOYED runtime.

    uv run pytest -m live tests/test_live_journey.py -q -s

This is a transcript somebody actually typed, not a scenario invented to pass.
It is here because every interesting failure in this build was found by a
person having an ordinary conversation, and none of them by a unit test:

  * "what are other benefits of this" was refused with "I could not find
    anything in our documented sources" - a citation refusal fired before
    anything read what the model had written, so the clarifying question it
    wrote was thrown away;
  * "what is copayment ?" was refused the same way, on a term the corpus
    covers in three documents.

The value of replaying a whole journey rather than the two failing turns is
that both needed the turns before them. A quote had been created, the thread
carried tool results, and "this" only means anything in that context.

**These assertions are about outcomes, not wording.** The model phrases
things differently every run - one run asks for "ages of family members",
the next for "who needs coverage" - so asserting on sentences would produce a
test that fails for the wrong reason. What is asserted is what a customer
would notice: an answer arrived, it was not a refusal, and where a state
change was proposed it was confirmed first.
"""
from __future__ import annotations

import json
import os
import time
import uuid

import pytest

pytestmark = pytest.mark.live

# The refusals this journey produced, verbatim from app/agents/citations.py.
# Matched on a distinctive fragment rather than the whole string so a reworded
# refusal still trips the test - the point is that a refusal happened.
REFUSAL_FRAGMENT = "could not find anything in our documented sources"
ACTION_REFUSAL_FRAGMENT = "I have not done that"
BLOCKED_FRAGMENT = "could not give you a reliable answer"
HANDOFF_FRAGMENT = "reasonable number of steps"


def _runtime_arn() -> str:
    arn = os.environ.get("AGENT_RUNTIME_ARN")
    if not arn:
        import boto3

        cf = boto3.client("cloudformation")
        out = cf.describe_stacks(StackName="mcb-agent")["Stacks"][0]["Outputs"]
        arn = next(o["OutputValue"] for o in out
                   if o["OutputKey"] == "AgentRuntimeArn")
    return arn


class Journey:
    """A conversation on one thread, with the glass box kept for each turn."""

    def __init__(self, user_id: str):
        import boto3

        self.user_id = user_id
        self.arn = _runtime_arn()
        self.client = boto3.client("bedrock-agentcore")
        self.turns: list[dict] = []

    def say(self, text: str) -> dict:
        r = self.client.invoke_agent_runtime(
            agentRuntimeArn=self.arn, qualifier="live",
            # A session id must be at least 33 characters, and a NEW one per
            # turn is deliberate: the thread carries the conversation, so a
            # fresh session proves the history came from the checkpoint and
            # not from a warm process.
            runtimeSessionId=uuid.uuid4().hex + uuid.uuid4().hex,
            payload=json.dumps({"messages": [{"role": "user",
                                              "content": text}],
                                "threadId": self.user_id}).encode())

        turn: dict = {"said": text, "reply": "", "events": [],
                      "tools": [], "error": None}
        for raw in r["response"].iter_lines():
            line = raw.decode() if isinstance(raw, bytes) else raw
            if not line.startswith("data:"):
                continue
            try:
                ev = json.loads(line[5:].strip())
            except ValueError:
                continue
            kind = ev.get("type")
            if kind == "TEXT_MESSAGE_CONTENT":
                turn["reply"] += ev.get("delta", "")
            elif kind == "RUN_ERROR":
                turn["error"] = json.dumps(ev)[:400]
            elif kind == "CUSTOM":
                v = ev.get("value") or {}
                turn["events"].append(v)
                if v.get("kind") == "tool":
                    turn["tools"].append(str(v.get("label") or ""))

        turn["reply"] = turn["reply"].strip()
        self.turns.append(turn)
        return turn

    @staticmethod
    def detail(turn: dict, label: str) -> dict:
        """The detail of the first event whose label CONTAINS `label`.

        Substring, not equality: a gate labels itself
        "confirmation_required quote_create_health", and matching that
        exactly made every lookup silently return {} - so the assertions
        built on it passed without checking anything.
        """
        for v in turn["events"]:
            if label in str(v.get("label") or ""):
                return v.get("detail") or {}
        return {}

    @staticmethod
    def retrieved(turn: dict) -> bool:
        """Did this turn go to the knowledge base at all?

        A follow-up often does not - the model answers from what the
        conversation already established, which the verifier allows. That
        distinction matters for citations: `[1]` means passage one OF THIS
        TURN, so a reference written without a retrieval resolves to nothing
        and is stripped.
        """
        return any(v.get("kind") == "retrieve" for v in turn["events"])

    @staticmethod
    def report(turn: dict) -> str:
        """What to print when an assertion fails: the trace, not just the
        reply. A refusal with an empty glass box and a refusal with a full one
        are different bugs."""
        lines = [f"  YOU: {turn['said']}"]
        for v in turn["events"]:
            label = str(v.get("label") or "")
            if v.get("kind") in ("guardrail", "gate", "retrieve", "tool"):
                lines.append(f"    {v.get('kind')}/{label}: "
                             f"{json.dumps(v.get('detail') or {})[:220]}")
        if turn["error"]:
            lines.append(f"    ERROR: {turn['error']}")
        lines.append(f"  BOT: {turn['reply'][:400]}")
        return "\n".join(lines)


def _answered(turn: dict) -> None:
    """The floor every turn has to clear."""
    assert turn["error"] is None, "the runtime errored\n" + Journey.report(turn)
    assert turn["reply"], "no reply at all\n" + Journey.report(turn)


def _not_refused(turn: dict, why: str) -> None:
    for fragment, name in ((REFUSAL_FRAGMENT, "the citation refusal"),
                           (ACTION_REFUSAL_FRAGMENT, "the action refusal"),
                           (BLOCKED_FRAGMENT, "the guardrail block message"),
                           (HANDOFF_FRAGMENT, "the tool round limit")):
        assert fragment not in turn["reply"], (
            f"{name} came back, and {why}\n" + Journey.report(turn))


@pytest.fixture(scope="module")
def journey():
    """A thread nobody has used before, so the journey starts from nothing.

    Consent for 'quotation' is granted first, because the entitlement gate
    refuses quote_create_health without it and that refusal would mask what
    this test is actually about. Granting it IS the customer journey - a
    person agreeing to a quote being prepared.
    """
    from app.memory.longterm import consent

    user_id = f"live-journey-{int(time.time())}"
    consent.record(user_id, "quotation", True,
                   {"source": "user_stated", "ref": "live journey test"})
    return Journey(user_id)


# ---------------------------------------------------------------------------
# the journey, in order - each test depends on the ones above it
# ---------------------------------------------------------------------------
def test_01_the_opening_question_asks_for_details(journey):
    """The turn that started all of this. It was refused once, because the
    citation rule saw the word "cover" in the bot's own question."""
    turn = journey.say("help me choose health insurance")
    _answered(turn)
    _not_refused(turn, "a question asking for details is not a claim")

    cites = Journey.detail(turn, "citations")
    assert cites.get("verifier") == "ran", (
        "the claim check did not run, so this answer had less scrutiny than "
        "it looks\n" + Journey.report(turn))
    assert cites.get("refused") is False


def test_02_a_city_on_its_own_is_understood(journey):
    turn = journey.say("pune")
    _answered(turn)
    _not_refused(turn, "naming a city is not a claim about cover")


def test_03_a_partial_answer_is_followed_up_not_refused(journey):
    """"myself, 1000000" gives the sum insured but not the age. Asking for
    the missing one is the correct move."""
    turn = journey.say("myself , 1000000")
    _answered(turn)
    _not_refused(turn, "asking for a missing detail is not a claim")


def _check_parked_quote(turn: dict) -> bool:
    """If a quote was proposed this turn, was it proposed HONESTLY?

    Returns whether one was parked. Not every run parks it on the same turn -
    the model sometimes lists the eligible products first - and asserting a
    turn number would make this fail for a reason that is not a fault. What
    must hold whenever a quote IS proposed: the arguments match what the
    customer actually said, and every one of them is visible in the question
    put to them. A confirmation that hides a field is not a confirmation of
    the call that will run.
    """
    gate = Journey.detail(turn, "quote_create_health")
    if not gate.get("token"):
        return False

    args = gate.get("args") or {}
    assert args.get("sum_insured") == 1000000, Journey.report(turn)
    assert str(args.get("city", "")).lower() == "pune", Journey.report(turn)
    assert args.get("member_ages") == [43], (
        "the age the customer just gave did not reach the tool call\n"
        + Journey.report(turn))
    for value in ("1000000", "43"):
        assert value in turn["reply"], (
            f"the confirmation hides {value!r}\n" + Journey.report(turn))
    assert "pune" in turn["reply"].lower(), Journey.report(turn)
    return True


def test_04_the_age_completes_the_picture(journey):
    """AG-6. A quote is a state change. Whether it is proposed on this turn
    or the next, it may not simply happen."""
    turn = journey.say("43")
    _answered(turn)
    _not_refused(turn, "giving an age is not a claim about cover")
    _check_parked_quote(turn)
    assert not Journey.detail(turn, "write quote_create_health"), (
        "a quote was created without being put to the customer first\n"
        + Journey.report(turn))


def test_05_yes_creates_the_quote(journey):
    """The consent path, read by a model rather than by a regex anchored on
    the first word of the reply.

    Says yes until the quote exists, at most twice: if the previous turn
    listed products rather than proposing one, the first yes proposes and the
    second confirms. Every proposal on the way is checked.
    """
    reply = "yes"
    for _ in range(4):
        turn = journey.say(reply)
        _answered(turn)
        _not_refused(turn, "consent was given for exactly what was proposed")

        # A CONFIRMED tool does not run through the tools node - the pending
        # node runs it directly, having held it since last turn, and traces
        # "write <tool>". Looking only at tool events missed the one path
        # this test exists to check.
        if Journey.detail(turn, "write quote_create_health") \
                or "quote_create_health" in turn["tools"]:
            answer = Journey.detail(turn, "confirmation_answer")
            assert answer.get("answer") == "yes", (
                "a plain yes was not read as consent\n"
                + Journey.report(turn))
            assert "Q-" in turn["reply"], (
                "the quote ran but no reference reached the customer\n"
                + Journey.report(turn))
            return

        if _check_parked_quote(turn):
            reply = "yes"                    # it proposed; confirm it
        else:
            # It asked something reasonable instead - most often which of the
            # three products to quote. Answering that is the journey, and a
            # test that only ever says "yes" would call a sensible question a
            # failure. Everything it could still be missing is restated,
            # which is what a customer asked twice actually does.
            reply = ("Protec Health Secure please, for myself, age 43, in "
                     "Pune, sum insured 10 lakh")

    pytest.fail("four turns in and still no quote\n"
                + Journey.report(journey.turns[-1]))


def test_06_a_vague_follow_up_is_answered_or_asked_about(journey):
    """The first reported failure.

    "What are other benefits of this" was refused with "I could not find
    anything in our documented sources". Retrieval had run and come back
    empty, and the refusal fired before anything read the reply - so the
    model's perfectly good "which plan did you mean?" was thrown away.

    Either outcome is acceptable here: answer from the documents, or ask
    which product is meant. What is not acceptable is a refusal.
    """
    turn = journey.say("what are other benefits of this")
    _answered(turn)
    _not_refused(turn, "a follow-up about the plan just quoted is ordinary")


def test_07_a_glossary_question_is_answered_from_the_documents(journey):
    """The second reported failure, and the more surprising one: co-payment
    is described in the brochure, the claim guide and the policy wording, and
    the question was still refused."""
    turn = journey.say("what is copayment ?")
    _answered(turn)
    _not_refused(turn, "co-payment is covered in three documents")

    cites = Journey.detail(turn, "citations")
    if Journey.retrieved(turn):
        assert cites.get("cited"), (
            "co-payment was looked up and answered with no document behind "
            "it\n" + Journey.report(turn))
    else:
        # Answered from what the conversation already established - the
        # product was described two turns ago. That is allowed, and it is why
        # a follow-up does not get refused for restating something the
        # customer was correctly told a moment before.
        assert "co-payment" in turn["reply"].lower() \
            or "copayment" in turn["reply"].lower(), Journey.report(turn)


def test_08_the_product_can_be_described(journey):
    """Blocked for weeks by a competitor-comparison topic that fired on the
    bot narrating its own search - "I will search for information about
    Health Secure in our approved sources" was refused 5 times out of 5."""
    turn = journey.say("meanwhile tell me more about health secure")
    _answered(turn)
    _not_refused(turn, "describing our own product is the job")

    cites = Journey.detail(turn, "citations")
    assert cites.get("cited"), (
        "the product was described with no document behind it\n"
        + Journey.report(turn))


def test_09_the_whole_journey_kept_its_citations_honest(journey):
    """Across every turn that retrieved: nothing cited was never sent.

    Only turns that retrieved. `[1]` means passage one OF THIS TURN, so when
    the model answers from what the conversation established and still writes
    a bracket, it resolves to nothing and is stripped - which is the rule
    working, not a fault. What would be a fault is a turn that retrieved four
    passages and cited a fifth.
    """
    invented = [(t["said"], Journey.detail(t, "citations").get("invented_refs"))
                for t in journey.turns
                if Journey.retrieved(t)
                and Journey.detail(t, "citations").get("invented_refs")]
    assert not invented, (
        "passages were cited that were never retrieved: " + repr(invented))

    ran = [t["said"] for t in journey.turns
           if Journey.detail(t, "citations").get("verifier") not in
           (None, "ran")]
    assert not ran, (
        "the claim check failed to run on: " + ", ".join(ran) +
        " - those answers had less scrutiny than the trace implies")
