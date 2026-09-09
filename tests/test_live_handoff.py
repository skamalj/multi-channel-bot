"""A handoff must not end the conversation.

    uv run pytest -m live tests/test_live_handoff.py -q -s

The rule, stated by the person testing it: a handoff should in no way impact
the conversation flow. The agent answers whatever comes next. If that means
saying "a colleague already has this", it says so; if the customer asks
something else, it answers that.

Two failures put this here, and the second was caused by the first.

  * A turn that ran out of tool rounds handed off, and the tool calls it had
    run out of budget for were never answered - an AI message carrying
    tool_calls left in the thread with no ToolMessage against it, which is
    the pair Bedrock rejects on the next turn.
  * The handoff left nothing the model could read: the customer-facing
    sentence is system_authored, so the model never saw it. On the next turn
    it invented a reason the conversation had stopped - "I have already
    searched the approved sources and no product information was found" -
    which was untrue, and which then poisoned every turn after it, because a
    checkpointed thread carries a false premise forward for as long as it
    lives.
"""
from __future__ import annotations

import time

import pytest

from tests.test_live_journey import Journey, _answered, _not_refused

pytestmark = pytest.mark.live

# Things the bot says when it has stopped trying, in one place so a reworded
# refusal still trips these tests.
GAVE_UP = ("could not find anything in our documented sources",
           "reasonable number of steps",
           "could not give you a reliable answer")


@pytest.fixture(scope="module")
def talk():
    return Journey(f"live-handoff-{int(time.time())}")


def test_1_a_product_question_is_answered(talk):
    turn = talk.say("tell me about health secure")
    _answered(turn)
    _not_refused(turn, "the product is documented")
    assert Journey.detail(turn, "citations").get("cited"), (
        "answered with no document behind it\n" + Journey.report(turn))


def test_2_the_customer_can_ask_for_a_person(talk):
    """Asking for a human is not something to argue with."""
    turn = talk.say("please connect me to a colleague")
    _answered(turn)


def test_3_the_conversation_continues_after_the_handoff(talk):
    """The turn straight after. It must not be a refusal, and it must not
    claim the sources were searched and came back empty - they were not."""
    turn = talk.say("while I wait, what is the waiting period for "
                    "pre-existing diseases?")
    _answered(turn)
    _not_refused(turn, "a handoff does not end the conversation")

    reply = turn["reply"].lower()
    for lie in ("no product information was found",
                "no information was found",
                "sources and no product"):
        assert lie not in reply, (
            "the bot claimed a search came back empty - it did not\n"
            + Journey.report(turn))


@pytest.mark.xfail(reason=(
    "KNOWN, and a decision rather than a bug to squash quietly. The model "
    "answered from what it had already retrieved and gave the PRE-AMENDMENT "
    "figure - pre-existing diseases at 48 months, which the corpus reduced "
    "to 36 from 1 April 2026 and still carries both versions of. The "
    "verifier dropped it, which is right. What was left was 'I can answer "
    "this from the information I already retrieved for you' - no refusal, no "
    "answer, just filler. Refusing when everything material is dropped would "
    "fix this and would break the case where a clarifying question is all "
    "that survives, which was the last thing fixed. Left visible until that "
    "trade-off is chosen deliberately."), strict=False)
def test_3b_and_the_answer_has_something_in_it(talk):
    """A customer asked a documented question and should get its answer.

    Separate from the test above on purpose: continuing the conversation and
    answering well are different claims, and only the first is fixed.
    """
    turn = talk.turns[-1]
    reply = turn["reply"].lower()
    assert "month" in reply or "36" in reply, (
        "the reply carries no answer at all\n" + Journey.report(turn))


def test_4_a_different_question_is_answered_not_deflected(talk):
    """The customer changing the subject is the ordinary case, not an edge
    one. Nothing about a queued handoff should stop it being answered."""
    turn = talk.say("what is copayment?")
    _answered(turn)
    _not_refused(turn, "co-payment is documented in three places")
    assert "co-pay" in turn["reply"].lower() \
        or "copay" in turn["reply"].lower(), Journey.report(turn)


def test_5_the_thread_survived_every_turn(talk):
    """The orphaned-tool-call check, end to end.

    A tool call left unanswered in the thread does not fail on the turn that
    made it - it fails on the NEXT one, when the whole history is replayed to
    Bedrock. So the evidence that the pairing held is simply that every turn
    after the handoff worked at all.
    """
    for turn in talk.turns:
        assert turn["error"] is None, (
            "the runtime rejected a turn, which is what an orphaned tool call "
            "looks like\n" + Journey.report(turn))
        assert turn["reply"], "a turn produced nothing\n" + Journey.report(turn)

    after = talk.turns[2:]
    assert after, "no turns after the handoff were exercised"
    for turn in after:
        assert not any(f in turn["reply"] for f in GAVE_UP), (
            "the bot gave up after the handoff\n" + Journey.report(turn))
