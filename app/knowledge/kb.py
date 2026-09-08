"""Retrieval from a Bedrock Knowledge Base, when one is deployed.

Semantic search only, top 5, no reranking (D-4). That is not a simplification
of the local retriever - it is a different instrument. S3 Vectors does not do
hybrid lexical matching at all, which is the price of a vector store with no
hourly floor, and it was an explicit decision rather than a discovery.

**What survives the move, and what does not.**

KB-2 survives, and it is the one that matters. `corpus_scope` and `lob` are
injected by the caller and become a metadata filter on the retrieve call, so
an agent-scoped document cannot be returned to a customer bot. That filter is
the reason `build_corpus.py` writes a `.metadata.json` sidecar for every
document: without it the scope would be a suggestion rather than a boundary.

KB-5 - the absolute score floor - and OB-4's rejected-candidate list do not
survive, because there is no local scoring pass to produce them. The safety
they provided moves to two places that already exist: Bedrock's contextual
grounding check, and `citations.py`, which will still refuse an answer whose
material claims do not resolve to a document that was actually retrieved.

The `Candidate` shape is preserved exactly so nothing downstream can tell the
difference. `bm25`, `cosine` and `coverage` are zero here and honestly so -
they were never computed, and reporting a plausible-looking number for them
would make the trace lie about how an answer was found.
"""
from __future__ import annotations

import logging

from app.config import settings
from app.knowledge.retriever import Candidate

log = logging.getLogger("mcb.kb")


def configured() -> bool:
    import os

    cfg = settings()
    if os.getenv("NO_AWS") == "1" or cfg.no_aws:
        return False
    return bool(cfg.knowledge_base_id)


def _client():
    import boto3

    return boto3.client("bedrock-agent-runtime",
                        region_name=settings().aws_region)


def _filter(lob: str, scopes: list[str], product: str | None) -> dict | None:
    """KB-2, as a server-side metadata filter.

    Scope is a list because a producer bot may read both the public corpus
    and the agent-only one. It is filtered HERE rather than after retrieval:
    post-filtering would mean asking for five results and sometimes keeping
    two, and the ones discarded would already have been read.
    """
    clauses: list[dict] = []
    if lob:
        clauses.append({"equals": {"key": "lob", "value": lob}})
    if scopes:
        clauses.append({"in": {"key": "scope", "value": list(scopes)}})
    if product:
        clauses.append({"equals": {"key": "product", "value": product}})
    if not clauses:
        return None
    return clauses[0] if len(clauses) == 1 else {"andAll": clauses}


def _candidate(result: dict, rank: int) -> Candidate:
    meta = result.get("metadata") or {}
    loc = (result.get("location") or {}).get("s3Location") or {}
    uri = loc.get("uri") or ""
    doc_id = meta.get("doc_id") or uri.rsplit("/", 1)[-1].removesuffix(".md")
    text = (result.get("content") or {}).get("text") or ""

    chunk = {
        # Stable and traceable to the retrieved passage. The citation
        # guardrail resolves ids against exactly this set.
        "chunk_id": f"{doc_id}#{rank}",
        "doc_id": doc_id,
        "text": text,
        "source": meta.get("source") or doc_id,
        # A Bedrock chunk has no section or page - it is a span of characters
        # chosen by the chunking strategy, not a structural unit. Saying so
        # is better than inventing a section number.
        "section": meta.get("section") or "",
        "page": meta.get("page") or 0,
        "authority": meta.get("authority") or "protec",
        "doc_type": meta.get("doc_type") or "",
        "product": meta.get("product") or "",
        "version": meta.get("version") or "",
        "effective_from": meta.get("effective_from") or "",
        "effective_to": meta.get("effective_to") or "",
        "scope": meta.get("scope") or "",
        "lob": meta.get("lob") or "",
        "title": meta.get("source") or doc_id,
        "date_sensitive": False,
    }
    score = float(result.get("score") or 0.0)
    return Candidate(chunk=chunk, score=score, fused=score,
                     reasons=["knowledge base, semantic"])


def search(query: str, lob: str, scopes: list[str], k: int | None = None,
           as_of: str | None = None, product: str | None = None
           ) -> tuple[list[Candidate], list[Candidate], dict]:
    """Same signature and same return shape as the local retriever.

    `rejected` is always empty: with no local ranking pass there is nothing
    that entered ranking and lost. Returning an empty list is the truthful
    answer, and the glass box says so rather than implying nothing was
    discarded when in fact nothing was ever scored.
    """
    cfg = settings()
    k = k or cfg.retrieval_k
    stats: dict = {"backend": "bedrock_kb", "kb_id": cfg.knowledge_base_id,
                   "k": k, "lob": lob, "scopes": list(scopes),
                   "candidates": 0, "accepted": 0, "rejected": 0}

    vector: dict = {"numberOfResults": k}
    flt = _filter(lob, scopes, product)
    if flt:
        vector["filter"] = flt
        stats["filter"] = flt

    try:
        resp = _client().retrieve(
            knowledgeBaseId=cfg.knowledge_base_id,
            retrievalQuery={"text": query},
            retrievalConfiguration={"vectorSearchConfiguration": vector})
    except Exception as exc:                                 # noqa: BLE001
        # Fails CLOSED. An empty result makes the agent say it could not find
        # anything and offer a colleague; falling back to the local corpus
        # would answer from a copy that may not match what was deployed.
        log.warning("kb retrieve failed: %s: %s", type(exc).__name__, exc)
        stats["error"] = type(exc).__name__
        return [], [], stats

    results = resp.get("retrievalResults") or []
    accepted = [_candidate(r, n) for n, r in enumerate(results, 1)]
    stats["candidates"] = len(results)
    stats["accepted"] = len(accepted)
    return accepted, [], stats
