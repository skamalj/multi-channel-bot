"""Citations (KB-4), the refusal path (KB-5) and confirmation (AG-6).

The citation rule was rebuilt after it failed in both directions in a single
turn: it refused the bot's own question because the word "cover" appeared in
it, and let "The premium is 12,499" through unchecked because a comma
defeated a number regex. The word lists are gone. Two mechanisms now:

* **code** decides whether `[2]` names a passage that was actually sent -
  array membership, tested here exhaustively because it must never be wrong;
* **a model** decides whether a sentence is a supported claim, and is
  stubbed in these tests. Its own behaviour is measured separately, against
  the real model, because a stub only proves the stub works.

Every failure that prompted the rebuild is a test below, so it stays fixed.
"""
import pytest

from app.agents import citations, confirm, verify

CHUNKS = [
    {"ref": 1, "chunk_id": "PHS-POLICY_WORDING-V2#1",
     "source": "Protec Health Secure policy wording V2", "text":
     "Pre-existing diseases are covered after 36 months of continuous cover. "
     "An initial waiting period of 30 days applies to all illnesses except "
     "accidental injury."},
    {"ref": 2, "chunk_id": "H-CLAIMS-PROC#1",
     "source": "Protec health claims procedure", "text":
     "Intimate a planned hospitalisation at least 48 hours before admission "
     "and an emergency within 24 hours of admission."},
]


@pytest.fixture
def verdict(monkeypatch):
    """Stub the verifier. Each test says what the model concluded."""
    def _set(unsupported=(), ran=True, error=None):
        monkeypatch.setattr(
            verify, "check",
            lambda *a, **k: verify.Verdict(
                ran=ran, unsupported=list(unsupported), error=error))
    return _set


# ---------------------------------------------------------------------------
# references - code, and it must never be wrong
# ---------------------------------------------------------------------------
def test_a_reference_to_a_passage_we_sent_survives(verdict):
    verdict()
    out, rep = citations.enforce(
        "Pre-existing diseases are covered after 36 months [1].", CHUNKS)
    assert "[1]" in out
    assert rep["cited"] == ["PHS-POLICY_WORDING-V2#1"]
    assert rep["refused"] is False


def test_a_reference_to_a_passage_we_never_sent_is_stripped(verdict):
    """A fake citation is worse than none - it looks like provenance."""
    verdict()
    out, rep = citations.enforce(
        "Your policy includes unlimited overseas cover [7].", CHUNKS)
    assert "[7]" not in out
    assert rep["invented_refs"] == [7]


def test_a_mixed_bracket_keeps_the_real_reference_and_drops_the_invented_one(verdict):
    verdict()
    out, rep = citations.enforce(
        "Pre-existing diseases are covered after 36 months [1, 9].", CHUNKS)
    assert "[1]" in out and "9" not in out.split("[1]")[1][:4]
    assert rep["invented_refs"] == [9]
    assert rep["cited"] == ["PHS-POLICY_WORDING-V2#1"]


def test_two_passages_in_one_bracket_is_a_normal_citation(verdict):
    """Citing two sources for one sentence is ordinary, and matching only
    single-reference brackets silently discarded grounded answers."""
    verdict()
    _, rep = citations.enforce(
        "Waiting periods and claim timelines are set out here [1, 2].", CHUNKS)
    assert rep["cited"] == ["PHS-POLICY_WORDING-V2#1", "H-CLAIMS-PROC#1"]


def test_the_audit_record_names_documents_not_positions(verdict):
    """`[2]` means nothing six months from now. The whole point of a citation
    is that somebody can go and read the thing."""
    verdict()
    _, rep = citations.enforce("Intimate within 24 hours [2].", CHUNKS)
    assert rep["cited"] == ["H-CLAIMS-PROC#1"]


def test_a_bracketed_aside_is_prose_not_a_failed_citation(verdict):
    verdict()
    out, rep = citations.enforce(
        "Cover applies after the waiting period [see your schedule].", CHUNKS)
    assert "see your schedule" in out
    assert rep["invented_refs"] == []


# ---------------------------------------------------------------------------
# the four failures that prompted the rebuild
# ---------------------------------------------------------------------------
def test_the_bot_may_ask_a_question_containing_the_word_cover(verdict):
    """The failure that started this. "Ages of family members you want to
    cover" was refused because a word list saw "cover" and called it a claim,
    so the bot could not ask for the details it needed to help."""
    verdict()                       # the model reports nothing unsupported
    answer = ("To help you choose, I need a few details:\n"
              "- Ages of family members you want to cover\n"
              "- Your city and pincode")
    out, rep = citations.enforce(answer, [], retrieval_ran=False)
    assert "you want to cover" in out
    assert rep["refused"] is False


def test_a_premium_written_with_a_comma_is_still_checked(verdict):
    """`The premium is 12,499` used to pass unchecked - the comma defeated
    the number regex - so a fabricated price reached the customer while a
    question about cover was refused."""
    verdict(unsupported=["The premium is 12,499 including GST."])
    out, rep = citations.enforce(
        "The premium is 12,499 including GST.", CHUNKS)
    assert "12,499" not in out
    assert rep["refused"] is True


def test_a_correct_answer_is_not_refused_for_mangling_an_identifier(verdict):
    """The model used to be handed `PHS-POLICY_WORDING-V2#1` and wrote
    `PHS-POLICYWORDING-V21`; a correct answer was refused on a string
    comparison. There is no fuzzy way to write `[1]`."""
    verdict()
    out, rep = citations.enforce(
        "Pre-existing diseases are covered after 36 months [1].", CHUNKS)
    assert rep["refused"] is False
    assert rep["cited"] == ["PHS-POLICY_WORDING-V2#1"]


def test_a_question_refuses_in_question_words_not_action_words(verdict):
    """A customer asking for help choosing was told "I have not done that",
    describing a transaction they never started. The wording is chosen by
    whether an ACTION ran, not by whether retrieval happened."""
    verdict(unsupported=["Cover is unlimited."])
    out, _ = citations.enforce("Cover is unlimited.", CHUNKS,
                               retrieval_ran=False, action_attempted=False)
    assert out == citations.REFUSAL

    out, _ = citations.enforce("Cover is unlimited.", CHUNKS,
                               retrieval_ran=True, action_attempted=True)
    assert out == citations.REFUSAL_ACTION


# ---------------------------------------------------------------------------
# refusal
# ---------------------------------------------------------------------------
def test_retrieval_that_found_nothing_refuses_a_claim(verdict):
    """With nothing retrieved there is nothing to cite, so the verifier flags
    the claim and the turn refuses."""
    verdict(unsupported=["Dental implants are covered."])
    out, rep = citations.enforce("Dental implants are covered.", [],
                                 retrieval_ran=True)
    assert out == citations.REFUSAL
    assert rep["refused"] is True


def test_retrieval_that_found_nothing_still_lets_the_bot_ask_a_question(verdict):
    """The failure this was reported as. "What are other benefits of this"
    scores below the retrieval floor, nothing comes back, and the model
    writes "which plan did you mean?" - which used to be thrown away and
    replaced with "I could not find anything in our documented sources".
    Refusing a question for lacking a citation is the same fault as refusing
    "ages of family members you want to cover"."""
    verdict()
    out, rep = citations.enforce(
        "Which product are you asking about - Health Secure, Super Top-Up or "
        "Senior Care?", [], retrieval_ran=True)
    assert "Which product" in out
    assert rep["refused"] is False


def test_without_a_verifier_an_empty_retrieval_still_refuses(verdict):
    """Degraded, and deliberately blunt. Nothing left can tell a question
    from a claim, so an unchecked answer with nothing behind it must not go
    out just because the model endpoint was slow."""
    verdict(ran=False, error="ThrottlingException")
    out, rep = citations.enforce("Dental implants are covered.", [],
                                 retrieval_ran=True)
    assert out == citations.REFUSAL
    assert rep["refused"] is True
    assert rep["verifier"] == "ThrottlingException"


def test_an_answer_whose_every_claim_is_unsupported_refuses(verdict):
    verdict(unsupported=["We give you a free gym membership.",
                         "You also get a cash bonus every year."])
    out, rep = citations.enforce(
        "We give you a free gym membership.\n"
        "You also get a cash bonus every year.", CHUNKS)
    assert out == citations.REFUSAL
    assert len(rep["dropped"]) == 2


def test_one_unsupported_sentence_is_dropped_and_the_rest_survives(verdict):
    verdict(unsupported=["We also give you a free gym membership."])
    out, rep = citations.enforce(
        "Pre-existing diseases are covered after 36 months [1].\n"
        "We also give you a free gym membership.", CHUNKS)
    assert "36 months" in out
    assert "gym membership" not in out
    assert rep["refused"] is False


def test_a_core_tool_result_stands_without_a_document(verdict):
    """A premium comes from the rating engine, not a brochure. The tool call
    is itself in the trace."""
    verdict()
    out, rep = citations.enforce("Your premium is Rs 24,780 plus GST.", [],
                                 other_tool_evidence=True)
    assert "24,780" in out and rep["refused"] is False


# ---------------------------------------------------------------------------
# the verifier failing is not the same as the answer passing
# ---------------------------------------------------------------------------
def test_an_unavailable_verifier_degrades_loudly_rather_than_silently(verdict):
    """The turn proceeds and the trace says verification did not run. The
    reference check still applied, so a fabricated citation was still
    removed - weaker, and honest about it."""
    verdict(ran=False, error="ThrottlingException")
    out, rep = citations.enforce(
        "Pre-existing diseases are covered after 36 months [1]. Also [7].",
        CHUNKS)
    assert rep["verifier"] == "ThrottlingException"
    assert rep["refused"] is False
    assert "[7]" not in out


def test_a_verdict_that_quotes_nothing_we_wrote_is_not_a_pass(verdict):
    """A verifier that reports unsupported claims we cannot locate has failed
    to do its job. Treating that as clean would pass exactly the answers it
    was most unsure about."""
    verdict(unsupported=["something the model never actually wrote"])
    out, rep = citations.enforce(
        "Pre-existing diseases are covered after 36 months [1].", CHUNKS)
    assert rep["refused"] is True


# ---------------------------------------------------------------------------
# confirmation
# ---------------------------------------------------------------------------
def test_the_confirmation_token_is_stable_across_the_round_trip():
    """AG-6. Keying on the message id would give the confirming turn a
    different key - which is exactly the retry that must not double-charge."""
    args = {"application_id": "A-1", "amount": 8000.0, "mode": "upi"}
    assert confirm.token_for("payment_collect", args) == \
        confirm.token_for("payment_collect", dict(reversed(list(args.items()))))
    assert confirm.token_for("payment_collect", args) != \
        confirm.token_for("payment_collect", args | {"amount": 9000.0})


# ---------------------------------------------------------------------------
# the verifier's own number-to-sentence mapping - code, so tested as code
# ---------------------------------------------------------------------------
@pytest.fixture
def verdict_json(monkeypatch):
    """Make the verifier run for real against a canned model reply."""
    from app.llm import bedrock

    def _set(payload: str):
        class _Msg:
            content = payload

        class _LLM:
            def invoke(self, _messages):
                return _Msg()

        monkeypatch.setattr(bedrock, "get_llm", lambda *a, **k: _LLM())
        monkeypatch.setattr(verify, "configured", lambda: True)
    return _set


ANSWER = ("Pre-existing diseases are covered after 36 months [1].\n"
          "We also give you a free gym membership.")


def test_a_number_names_the_sentence_it_points_at(verdict_json):
    verdict_json('{"unsupported": [2]}')
    v = verify.check(ANSWER, CHUNKS)
    assert v.ran and v.unsupported == ["We also give you a free gym membership."]


def test_a_number_outside_the_draft_is_not_a_sentence(verdict_json):
    """A verdict naming sentence 9 of a two-sentence reply located nothing.
    Treating that as clean would pass the answers it was least sure about."""
    verdict_json('{"unsupported": [9]}')
    assert verify.check(ANSWER, CHUNKS).ran is False


def test_a_verdict_naming_nothing_is_a_pass(verdict_json):
    verdict_json('{"unsupported": []}')
    v = verify.check(ANSWER, CHUNKS)
    assert v.ran and v.unsupported == []


def test_numbers_arriving_as_strings_still_map(verdict_json):
    """Models write "2." as readily as 2, and that is not a failed check."""
    verdict_json('{"unsupported": ["2."]}')
    assert verify.check(ANSWER, CHUNKS).unsupported == \
        ["We also give you a free gym membership."]


def test_a_reply_that_is_not_json_is_a_failed_check_not_a_pass(verdict_json):
    verdict_json("I could not determine that.")
    v = verify.check(ANSWER, CHUNKS)
    assert v.ran is False and v.error == "unparseable_verdict"


# The Bedrock-guardrail tests that used to sit here are gone with the feature.
# Guardrails are provider-agnostic middleware now (app/agents/guardrails.py,
# grounding.py); the input screen and grounding are exercised through the
# agent, and there is no `guardrails=` on the model object to assert about.


# ---------------------------------------------------------------------------
# the input guardrail and the PII mask - provider-agnostic middleware
# ---------------------------------------------------------------------------
def test_the_pii_mask_hides_a_full_number_but_keeps_the_last_four():
    """The one sanctioned deterministic guardrail: pattern-matching digit
    groups, not a language judgement. A full Aadhaar or card the model echoed
    is masked at the output boundary; the last four survive for recognition."""
    from app.agents.guardrails import mask_pii

    out = mask_pii("Your card 4111 1111 1111 1234 is on file.")
    assert "4111 1111 1111 1234" not in out and "1234" in out
    out = mask_pii("Aadhaar 1234 5678 9012 linked.")
    assert "1234 5678 9012" not in out and "9012" in out
    # A short number - a PIN code, an amount - is left alone.
    assert mask_pii("Premium is 12499 in Pune 411001.") == \
        "Premium is 12499 in Pune 411001."


def test_the_input_guardrail_is_off_without_a_model(monkeypatch):
    """Offline and in tests there is no model to ask, so it never blocks - the
    same posture as the verifier."""
    from app.agents import guardrails

    assert guardrails.configured() is False   # MOCK_LLM / NO_AWS in tests


# ---------------------------------------------------------------------------
# consent the customer can actually give
# ---------------------------------------------------------------------------
def test_a_consent_denial_tells_the_model_to_ask_rather_than_apologise():
    """Reported: "I'm unable to generate a quote at this moment due to a
    system authorization issue."

    Every quote tool needs consent for 'quotation', no seeded customer had
    it, and nothing in the conversation could grant it - so the customer met
    a wall described as a fault. Needing permission is not the same as not
    being allowed.
    """
    from app.agents.context import RequestContext
    from app.agents.controls import _authorize, _refusal
    from app.mcpserver.registry import get_tool

    extras = get_tool("quote_create_health").extras or {}
    ctx = RequestContext(persona="customer", lob="health", authenticated=True,
                         customer_id="C-10001", user_id="u-consent", consent={})
    ok, why = _authorize(extras, ctx, {})
    assert not ok and "consent" in why

    out = _refusal(extras, why, runtime=None)
    assert out["error"] == "consent_required", out
    assert out["purpose"] == "quotation"
    assert "consent_grant" in out["try_instead"]
    assert "fault" in out["remedy"]


def test_consent_is_recorded_only_for_a_purpose_that_exists():
    from app.agents.context import RequestContext
    from app.mcpserver.tools.policy import consent_grant

    def rt(uid):
        return type("R", (), {"context": RequestContext(user_id=uid)})()

    bad = consent_grant.func(purpose="everything", runtime=rt("u-consent"))
    assert bad["error"] == "unknown_purpose", bad
    assert "quotation" in bad["allowed"]

    nobody = consent_grant.func(purpose="quotation", runtime=rt(None))
    assert nobody["error"] == "no_subject"


def test_granting_consent_unblocks_the_call_that_needed_it():
    """End to end through the ledger: the gate refuses, consent is given,
    the gate passes."""
    from app.agents.context import RequestContext
    from app.agents.controls import _authorize
    from app.memory.longterm import consent as ledger
    from app.mcpserver.registry import get_tool
    from app.mcpserver.tools.policy import consent_grant

    extras = get_tool("quote_create_health").extras or {}
    ctx = RequestContext(persona="customer", lob="health", authenticated=True,
                        customer_id="C-1", user_id="u-consent-flow", consent={})
    ok, why = _authorize(extras, ctx, {})
    assert not ok and "consent" in why

    # consent_grant reads the subject from runtime.context; drive its body.
    consent_grant.func(purpose="quotation",
                       runtime=type("R", (), {"context": ctx})())
    ctx.consent = ledger.current("u-consent-flow")
    assert ctx.consent.get("quotation") is True

    ok, why = _authorize(extras, ctx, {})
    assert ok, why


def test_consent_is_confirmed_before_it_is_recorded():
    """AG-6 applies to consent more than to anything else: it is a write, so
    the customer is shown the purpose and nothing is written until they say
    yes. Their yes IS the consent."""
    from app.mcpserver.registry import get_tool

    assert (get_tool("consent_grant").extras or {})["effect"] == "write"


def test_the_idempotency_token_is_stable_across_the_round_trip():
    """All that survives of the confirmation machinery, and the one part a
    prompt cannot do: the same call agreed to twice writes once.

    Keying on the message id would give the agreeing turn a different key,
    which is exactly the retry that must not double-charge.
    """
    args = {"application_id": "A-1", "amount": 8000.0, "mode": "upi"}
    assert confirm.token_for("payment_collect", args) == \
        confirm.token_for("payment_collect", dict(reversed(list(args.items()))))
    assert confirm.token_for("payment_collect", args) != \
        confirm.token_for("payment_collect", args | {"amount": 9000.0})


def test_a_tool_documents_its_parameters_not_the_workflow():
    """A description says what the tool IS and what each parameter means.
    What to do before calling it is workflow, and workflow is in the prompt -
    one place that describes the journey, rather than a rule repeated across
    twelve tool descriptions and impossible to read as a whole.
    """
    from langchain_core.utils.function_calling import convert_to_openai_tool

    from app.mcpserver.registry import get_tool

    quote = convert_to_openai_tool(get_tool("quote_create_health"))["function"]
    schema = quote["parameters"]
    props = schema["properties"]

    # A description says what a type cannot - a sum insured is rupees, not lakhs.
    assert "rupees" in props["sum_insured"]["description"].lower()
    # Required vs optional is the schema's own `required` list, not prose.
    assert "sum_insured" in schema["required"]
    assert "addons" not in schema["required"]
    # Injected args (the runtime, entitlements) are not in the model's schema.
    assert "runtime" not in props
    # And no workflow smuggled into the description.
    assert "wait" not in quote["description"].lower()


def test_the_prompt_names_the_tools_that_change_something():
    """The model has to know which calls need asking about. That list is
    workflow, so it lives in the prompt with the rest of the journey."""
    from app.agents.prompts import load

    text = load("health_customer.md")
    for name in ("quote_create_health", "payment_collect", "policy_issue",
                 "consent_grant"):
        assert name in text, f"the prompt never names {name}"
