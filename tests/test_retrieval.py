"""Retrieval: the pre-filter is structural, the floor is real, and the
rejects are visible."""
from app.knowledge.corpus import stats
from app.knowledge.retriever import prefilter, search


def test_the_corpus_is_generated_and_carries_the_declared_metadata():
    """KB-7 and KB-1. Every chunk, every field, no exceptions."""
    from app.knowledge.corpus import corpus

    rows = corpus()
    assert stats()["documents"] >= 70
    required = {"chunk_id", "doc_id", "product", "lob", "doc_type",
                "authority", "scope", "version", "effective_from",
                "effective_to", "section", "page", "source", "text"}
    for c in rows:
        assert required <= set(c), f"{c['chunk_id']} is missing metadata"


def test_no_two_documents_disagree_about_a_figure():
    """KB-7's whole point: the brochure and the wording render from one YAML.

    The current PHS wording says 36 months; so must every other current
    document that mentions it.
    """
    from app.coremock.catalog import product
    from app.knowledge.corpus import corpus

    months = product("PHS")["waiting_periods"]["ped_months"]
    mentions = [c for c in corpus()
                if c["product"] == "PHS" and c["version"] == "V2"
                and "pre-existing" in c["text"].lower()
                and "month" in c["text"].lower()]
    assert mentions
    for c in mentions:
        assert f"{months} months" in c["text"], c["chunk_id"]


def test_scope_is_filtered_before_ranking_not_after():
    """KB-2. An agent-only document is not ranked lower for a customer bot,
    it is not a candidate at all."""
    keep, dropped = prefilter("motor", ["public"])
    assert dropped["scope"] > 0
    accepted, rejected, _ = search("commission grid payout", "motor",
                                   ["public"])
    seen = {c.chunk["chunk_id"] for c in accepted + rejected}
    assert not any(cid.startswith("M-COMM-GRID") for cid in seen)

    accepted2, _, _ = search("commission grid payout", "motor",
                             ["public", "agent"])
    assert any(c.chunk["doc_id"] == "M-COMM-GRID" for c in accepted2)


def test_the_other_line_of_business_is_never_a_candidate():
    accepted, rejected, st = search("waiting period", "motor",
                                    ["public", "agent"])
    assert all(c.chunk["lob"] == "motor" for c in accepted + rejected)
    assert st["dropped"]["lob"] > 0


def test_an_unanswerable_question_returns_nothing():
    """KB-5. An empty result is a correct answer, and the caller refuses."""
    accepted, rejected, st = search("do you insure pizza delivery on mars",
                                    "health", ["public"])
    assert accepted == []
    assert st["accepted"] == 0
    assert all(c.rejected for c in rejected)


def test_rejected_candidates_come_back_with_their_scores():
    """OB-4. A retrieval you cannot see the misses of is one you cannot tune."""
    accepted, rejected, _ = search("waiting period", "health", ["public"])
    assert accepted and rejected
    for c in rejected:
        d = c.as_dict(with_text=False)
        assert d["rejected"]
        assert {"bm25", "cosine", "score", "section"} <= set(d)


def test_a_policy_dated_question_sees_only_the_wording_in_force_then():
    """KB-6. Two versions of one wording, answered per policy date."""
    old_keep, _ = prefilter("health", ["public"], as_of="2025-06-01")
    new_keep, _ = prefilter("health", ["public"], as_of="2026-06-01")

    from app.knowledge.retriever import _index

    idx = _index()
    old_ids = {idx.chunks[i]["doc_id"] for i in old_keep}
    new_ids = {idx.chunks[i]["doc_id"] for i in new_keep}
    assert "PHS-POLICY_WORDING-V1" in old_ids
    assert "PHS-POLICY_WORDING-V2" not in old_ids
    assert "PHS-POLICY_WORDING-V2" in new_ids
    assert "PHS-POLICY_WORDING-V1" not in new_ids
    # A 2026 circular is not the answer to a question about a 2025 policy.
    assert "H-PED-AMEND" not in old_ids and "H-PED-AMEND" in new_ids


def test_undated_process_documents_stay_retrievable_on_any_date():
    """Only versioned documents are filtered BY date; a claims process is not
    hidden because the question is about an older policy."""
    keep, _ = prefilter("health", ["public"], as_of="2025-06-01")
    from app.knowledge.retriever import _index

    ids = {_index().chunks[i]["doc_id"] for i in keep}
    assert "H-CLAIMS-PROC" in ids


def test_metadata_modulates_relevance_it_does_not_manufacture_it():
    """A chunk with the right doc_type and no term overlap must not surface.

    Normalising scores to the best hit in the query would make the best of
    five bad chunks score 1.0, and the floor would never fire.
    """
    accepted, _, _ = search("qwertyuiop zxcvbnm", "health", ["public"])
    assert accepted == []
