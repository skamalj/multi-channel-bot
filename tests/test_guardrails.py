"""Citations (KB-4), the refusal path (KB-5) and confirmation (AG-6)."""
from app.agents import citations, confirm

CHUNKS = [
    {"chunk_id": "PHS-POLICY_WORDING-V2#1", "text":
     "Pre-existing diseases are covered after 36 months of continuous cover. "
     "An initial waiting period of 30 days applies to all illnesses except "
     "accidental injury."},
    {"chunk_id": "H-CLAIMS-PROC#1", "text":
     "Intimate a planned hospitalisation at least 48 hours before admission "
     "and an emergency within 24 hours of admission."},
]


def test_a_cited_sentence_survives():
    out, rep = citations.enforce(
        "Pre-existing diseases are covered after 36 months "
        "[PHS-POLICY_WORDING-V2#1].", CHUNKS)
    assert "36 months" in out
    assert rep["cited"] == ["PHS-POLICY_WORDING-V2#1"]
    assert rep["dropped"] == []


def test_an_uncited_claim_is_dropped_before_the_customer_sees_it():
    out, rep = citations.enforce(
        "Pre-existing diseases are covered after 36 months "
        "[PHS-POLICY_WORDING-V2#1]. We also give you a free gym membership "
        "and a cash bonus every year you do not claim.", CHUNKS)
    assert "gym membership" not in out
    assert rep["dropped"] and "gym" in rep["dropped"][0]


def test_a_forgotten_citation_on_a_grounded_sentence_is_repaired_not_deleted():
    """The sentence was grounded; the formatting was not. Deleting a correct
    sentence is the worse error."""
    out, rep = citations.enforce(
        "Intimate a planned hospitalisation at least 48 hours before "
        "admission and an emergency within 24 hours of admission.", CHUNKS)
    assert "[H-CLAIMS-PROC#1]" in out
    assert rep["repaired"] and rep["dropped"] == []


def test_a_hallucinated_chunk_id_is_stripped_and_the_claim_is_dropped():
    """A fake marker is worse than none - it looks like provenance."""
    out, rep = citations.enforce(
        "Your policy includes unlimited overseas cover [PHS-PROSP-9.9].",
        CHUNKS)
    assert "PHS-PROSP-9.9" not in out
    assert rep["hallucinated"] == ["PHS-PROSP-9.9"]
    assert rep["refused"] is True


def test_conversational_sentences_do_not_need_a_source():
    out, _ = citations.enforce(
        "Pre-existing diseases are covered after 36 months "
        "[PHS-POLICY_WORDING-V2#1]. Shall I check your policy for you?",
        CHUNKS)
    assert "Shall I check your policy for you?" in out


def test_empty_retrieval_refuses_and_offers_a_human():
    """KB-5. An empty answer is a correct answer."""
    out, rep = citations.enforce("The waiting period is two weeks.", [])
    assert rep["refused"] is True
    assert "colleague" in out


def test_a_core_tool_result_is_evidence_even_without_a_chunk():
    """A premium comes from the rating engine, and the tool call is in the
    trace - it does not need a document citation."""
    out, rep = citations.enforce(
        "Your premium is Rs 24,780 plus GST.", [], other_tool_evidence=True)
    assert "24,780" in out and rep["refused"] is False


# --- confirmation ----------------------------------------------------------
def test_the_confirmation_token_is_stable_across_the_round_trip():
    """AG-6. Keying on the message id would give the confirming turn a
    different key - which is exactly the retry that must not double-charge."""
    args = {"application_id": "A-1", "amount": 8000.0, "mode": "upi"}
    assert confirm.token_for("payment_collect", args) == \
        confirm.token_for("payment_collect", dict(reversed(list(args.items()))))
    assert confirm.token_for("payment_collect", args) != \
        confirm.token_for("payment_collect", args | {"amount": 9000.0})


def test_the_summary_names_every_argument():
    """A confirmation that hides a field is not a confirmation of the call
    that will run."""
    summary = confirm.summarise("payment_collect",
                                {"application_id": "A-1", "amount": 8000.0,
                                 "mode": "upi", "_idempotency_key": "x"})
    assert "8000" in summary and "upi" in summary and "A-1" in summary
    assert "_idempotency_key" not in summary


def test_an_unclear_answer_is_not_a_yes():
    assert confirm.read_answer("yes please") == "yes"
    assert confirm.read_answer("no thanks") == "no"
    assert confirm.read_answer("what does that cost?") == "unclear"
    assert confirm.read_answer("") == "unclear"


def test_a_parked_confirmation_expires():
    import time

    state: dict = {}
    confirm.park(state, "policy_issue", {"application_id": "A-1"})
    assert confirm.pending_of(state)
    state["pending_confirmation"]["ts"] = time.time() - confirm.TTL_S - 1
    assert confirm.pending_of(state) is None
    assert "pending_confirmation" not in state


def test_headings_and_hedges_survive_but_invented_figures_do_not():
    """The naive rule - a citation per sentence - deletes the shape of the
    answer and leaves something safe and unreadable. The claims that hurt
    somebody are the specific ones."""
    answer = (
        "**Waiting periods on your plan:**\n"
        "Pre-existing diseases are covered after 36 months of continuous "
        "cover [PHS-POLICY_WORDING-V2#1].\n"
        "It depends on when your policy started.\n"
        "You also get unlimited free international treatment.\n"
        "Would you like me to check your own policy?")
    out, rep = citations.enforce(answer, CHUNKS)
    assert "Waiting periods on your plan" in out          # heading kept
    assert "It depends on when your policy started" in out  # hedge kept
    assert "Would you like me to check your own policy?" in out
    assert "unlimited free international treatment" not in out
    assert rep["dropped"] and "unlimited" in rep["dropped"][0]


def test_a_bold_wrapped_question_is_not_treated_as_a_claim():
    out, _ = citations.enforce(
        "Pre-existing diseases are covered after 36 months "
        "[PHS-POLICY_WORDING-V2#1].\n"
        "**Would you like me to check the exact dates for your policy?**",
        CHUNKS)
    assert "check the exact dates" in out


def test_an_answer_of_nothing_but_hedging_still_refuses():
    """Keeping prose must not become a way to answer with no substance."""
    out, rep = citations.enforce(
        "It depends on your policy. I can look that up for you.", CHUNKS)
    assert rep["refused"] is True and "colleague" in out


# --- figures behind a core tool result ------------------------------------
QUOTE_FACTS = ('{"product_name": "Protec Motor Shield", "idv": 480000, '
               '"od_net": 10657.92, "tp_premium": 2094, "cpa_premium": 330, '
               '"ncb_pct": 35, "gross_premium": 16682.75, '
               '"quote_id": "Q-9D57E12C", "valid_until": "2026-09-19"}')


def test_a_premium_from_the_core_survives_with_its_own_figures():
    out, rep = citations.enforce(
        "Your total premium is Rs 16,682.75 including 18% GST, and the IDV "
        "is Rs 4,80,000.", [], other_tool_evidence=True,
        tool_facts=QUOTE_FACTS + ' {"gst_pct": 18}')
    assert "16,682.75" in out and rep["dropped"] == []


def test_a_figure_the_core_never_returned_is_dropped():
    """One true sentence about the premium must not license every other
    sentence in the same answer. This is the drift that reached the console:
    a vehicle year restated from the model's own earlier prose."""
    out, rep = citations.enforce(
        "Your total premium is Rs 16,682.75 including GST.\n"
        "Vehicle: MH12AB1234 (Maruti Swift VXI, 2021, Petrol).",
        [], other_tool_evidence=True, tool_facts=QUOTE_FACTS)
    assert "16,682.75" in out
    assert "2021" not in out
    assert rep["dropped"] and "Swift" in rep["dropped"][0]


def test_an_identifier_is_not_read_as_three_numbers():
    """"Q-9D57E12C" is a quote id, not the figures 9, 57 and 12 - reading it
    as figures would reject a quote id for not appearing in its own quote."""
    out, rep = citations.enforce(
        "Your quote reference is Q-9D57E12C and it is valid until "
        "19 September 2026.", [], other_tool_evidence=True,
        tool_facts=QUOTE_FACTS)
    assert "Q-9D57E12C" in out and rep["dropped"] == []


def test_a_sensibly_rounded_figure_is_still_that_figure():
    out, rep = citations.enforce(
        "Your total premium is about Rs 16,683.", [], other_tool_evidence=True,
        tool_facts=QUOTE_FACTS)
    assert "16,683" in out and rep["dropped"] == []


def test_two_sources_in_one_bracket_is_a_normal_citation():
    """"[A#1, B#2]" is how a model cites two sources for one sentence.

    Matching only single-id brackets silently discarded correctly grounded
    answers - the sentence looked uncited, failed the overlap repair, and the
    whole turn refused.
    """
    out, rep = citations.enforce(
        "Pre-existing diseases are covered after 36 months "
        "[PHS-POLICY_WORDING-V2#1, H-CLAIMS-PROC#1].", CHUNKS)
    assert rep["refused"] is False
    assert set(rep["cited"]) == {"PHS-POLICY_WORDING-V2#1", "H-CLAIMS-PROC#1"}
    assert rep["dropped"] == []


def test_a_mixed_bracket_keeps_the_real_id_and_strips_the_invented_one():
    out, rep = citations.enforce(
        "Pre-existing diseases are covered after 36 months "
        "[PHS-POLICY_WORDING-V2#1, PHS-MADE-UP#9].", CHUNKS)
    assert "PHS-POLICY_WORDING-V2#1" in out
    assert "PHS-MADE-UP#9" not in out
    assert rep["hallucinated"] == ["PHS-MADE-UP#9"]
    assert rep["refused"] is False


def test_a_bracketed_aside_is_prose_not_a_failed_citation():
    out, rep = citations.enforce(
        "Pre-existing diseases are covered after 36 months "
        "[PHS-POLICY_WORDING-V2#1]. See the table [above] for the rest.",
        CHUNKS)
    assert "[above]" in out
    assert rep["hallucinated"] == []


def test_a_numbered_list_of_questions_is_not_a_list_of_claims():
    """A list marker is formatting, not a figure.

    Left in, "1. Your registration number" is a sentence containing a digit,
    so a bot asking two clarifying questions in a numbered list looked like
    it was making two unsourced claims - and the whole turn refused.
    """
    answer = ("To generate your quote, I need to know:\n"
              "1. Your car's registration number - this helps me look up "
              "your vehicle details\n"
              "2. When is your current policy expiring? - so there is no gap")
    out, rep = citations.enforce(answer, [], other_tool_evidence=False)
    assert rep["refused"] is False
    assert "registration number" in out
    assert rep["dropped"] == []


def test_a_numbered_list_of_invented_figures_is_still_dropped():
    """The marker is stripped; the claim underneath is still checked."""
    out, rep = citations.enforce(
        "1. Your policy includes unlimited overseas cover worth Rs 50,00,000",
        [], other_tool_evidence=False)
    assert rep["refused"] is True


# --- transactional turns ---------------------------------------------------
def test_an_identifier_is_not_a_material_claim():
    """"Any digit" was the first version of the materiality test, and it made
    a bot walking a customer through issuance refuse its way out of its own
    journey: an application id and a numbered step both contain digits."""
    answer = ("Your application has been started.\n"
              "Application ID: A-47C234A2\n"
              "To complete your renewal we need three steps:\n"
              "1. KYC verification\n"
              "2. Pre-inspection\n"
              "3. Payment")
    out, rep = citations.enforce(answer, [], other_tool_evidence=False)
    assert rep["refused"] is False
    assert "A-47C234A2" in out and "Pre-inspection" in out


def test_a_figure_this_journey_established_is_a_reference_not_a_claim():
    """The core returned the quote total two turns ago. Quoting it back is
    not a new claim about price."""
    out, rep = citations.enforce(
        "Your quote Q-9D57E12C for Rs 16,682.75 is still valid until "
        "19 September 2026.", [], other_tool_evidence=False,
        tool_facts=QUOTE_FACTS)
    assert rep["refused"] is False
    assert "16,682.75" in out


def test_a_promise_is_never_rescued_by_an_established_figure():
    """A promise has no figure to check, so nothing but a cited source can
    support it - otherwise one real quote licenses any claim about cover."""
    out, rep = citations.enforce(
        "Your quote Q-9D57E12C also includes unlimited overseas cover.",
        [], other_tool_evidence=False, tool_facts=QUOTE_FACTS)
    assert rep["refused"] is True


def test_an_action_turn_refuses_in_action_words():
    """A knowledge-shaped refusal told to a customer who asked to issue a
    policy sounds like a knowledge gap, when what happened is that nothing
    was done."""
    knowledge, _ = citations.enforce(
        "The waiting period is two weeks.", [], retrieval_ran=True)
    action, _ = citations.enforce(
        "I have issued your policy, covered from today.", [],
        retrieval_ran=False)
    assert "documented sources" in knowledge
    assert "have not done that" in action


def test_a_fabricated_gate_decision_is_refused():
    """The worst sentence in the journey, and it has no figure in it.

    "Underwriting Decision: CLEARED" written without calling underwriting
    tells the customer the most consequential thing there is. The gate caught
    it two turns later - by which point they had been told.
    """
    out, rep = citations.enforce(
        "Your application has been sent to underwriting.\n"
        "Underwriting Decision: CLEARED", [], other_tool_evidence=False,
        tool_facts=QUOTE_FACTS)
    assert rep["refused"] is True
    assert "CLEARED" not in out


def test_a_real_gate_decision_from_a_tool_still_reads_normally():
    """The same words, behind an actual tool result, must survive - otherwise
    the bot cannot report the outcome it just obtained."""
    facts = '{"application_id": "A-1", "gate": "uw", "state": "cleared"}'
    out, rep = citations.enforce(
        "Underwriting is cleared for application A-1.\n"
        "The remaining gate is payment.", [], other_tool_evidence=True,
        tool_facts=facts)
    assert rep["refused"] is False
    assert "cleared" in out


VERSIONED = [
    {"chunk_id": "PHS-POLICY_WORDING-V2#1",
     "text": "Pre-existing diseases are covered after 36 months."},
    {"chunk_id": "PHST-POLICY_WORDING#1",
     "text": "Pre-existing diseases are covered after 24 months."},
]


def test_a_versionless_citation_resolves_to_the_chunk_it_meant():
    """Only PHS has dated wordings, so its ids carry -V2 and every other
    product's do not. A model reading both writes the versionless form; that
    is a typo pointing at a chunk we retrieved, not an invented source."""
    out, rep = citations.enforce(
        "Pre-existing diseases are covered after 36 months "
        "[PHS-POLICY_WORDING#1].", VERSIONED)
    assert rep["hallucinated"] == []
    assert rep["cited"] == ["PHS-POLICY_WORDING-V2#1"]
    assert "PHS-POLICY_WORDING-V2#1" in out


def test_an_ambiguous_versionless_citation_is_not_resolved():
    """If both wordings were retrieved the model has not said which one it
    means - and that distinction is the whole of KB-6."""
    both = VERSIONED + [{"chunk_id": "PHS-POLICY_WORDING-V1#1",
                         "text": "Pre-existing diseases are covered after "
                                 "48 months."}]
    _, rep = citations.enforce(
        "Pre-existing diseases are covered after 36 months "
        "[PHS-POLICY_WORDING#1].", both)
    assert rep["hallucinated"] == ["PHS-POLICY_WORDING#1"]


def test_a_genuinely_invented_id_is_still_a_hallucination():
    _, rep = citations.enforce(
        "Your plan includes overseas cover [PHS-OVERSEAS-V9#3].", VERSIONED)
    assert rep["hallucinated"] == ["PHS-OVERSEAS-V9#3"]
