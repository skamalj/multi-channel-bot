"""Retrieval: pre-filter, hybrid search, rerank, floor.

Four things happen in this order, and the order is the design:

1. **Metadata pre-filter (KB-2).** `lob` and `scope` are applied BEFORE any
   ranking. Entitlement is structural: an agent-only commission grid is not
   "ranked lower" for a customer bot, it is not a candidate. Effective dating
   is applied here too, so a question about a 2025 policy never sees the 2026
   wording (KB-6).
2. **Hybrid retrieval (KB-3).** BM25 for lexical precision and a TF-IDF
   cosine for term-overlap recall, fused with reciprocal rank fusion. Two
   weak, cheap signals that fail differently beat one signal.
3. **Rerank and coverage (KB-3).** A deterministic feature reranker -
   section-title hit, exact phrase, authority, document-type affinity,
   recency - modulating an absolute lexical/semantic score, multiplied by how
   much of the QUESTION's information the chunk actually matches. Optionally
   a small Bedrock model, which is the drop-in for a real cross-encoder.
4. **Floor (KB-5).** Below the floor, nothing is returned and the caller
   refuses and offers a human. An empty result is a correct answer.

Every candidate that was considered and dropped is returned as well, with
its scores and the reason (OB-4). The console renders the rejects; a
retrieval you cannot see the misses of is a retrieval you cannot tune.
"""
from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from datetime import date
from functools import lru_cache

from app.config import settings
from app.knowledge.corpus import corpus

_STOP = {
    "the", "and", "for", "are", "was", "with", "that", "this", "from", "have",
    "has", "not", "but", "you", "your", "our", "can", "will", "what", "when",
    "how", "why", "does", "did", "any", "all", "its", "it's", "about", "into",
    "than", "then", "there", "their", "they", "them", "who", "whom", "been",
    "under", "over", "per", "may", "must", "shall", "would", "could", "should",
}


def _tokens(text: str) -> list[str]:
    cleaned = "".join(
        c if c.isalnum() else " " for c in (text or "").lower())
    return [w for w in cleaned.split() if len(w) > 2 and w not in _STOP]


@dataclass
class Candidate:
    chunk: dict
    bm25: float = 0.0
    cosine: float = 0.0
    coverage: float = 0.0
    fused: float = 0.0
    rerank: float = 0.0
    score: float = 0.0
    reasons: list[str] = field(default_factory=list)
    rejected: str | None = None

    def as_dict(self, with_text: bool = True) -> dict:
        c = self.chunk
        out = {
            "chunk_id": c["chunk_id"], "score": round(self.score, 3),
            "bm25": round(self.bm25, 3), "cosine": round(self.cosine, 3),
            "fused": round(self.fused, 4), "rerank": round(self.rerank, 3),
            "coverage": round(self.coverage, 2),
            "source": c["source"], "section": c["section"],
            "page": c["page"], "authority": c["authority"],
            "doc_type": c["doc_type"], "product": c["product"],
            "version": c["version"],
            "effective_from": c["effective_from"],
            "why": ", ".join(self.reasons),
        }
        if self.rejected:
            out["rejected"] = self.rejected
        if with_text:
            out["text"] = c["text"]
        return out


# ---------------------------------------------------------------------------
# Index. Built once per process from the corpus.
# ---------------------------------------------------------------------------
class _Index:
    def __init__(self, chunks: list[dict]):
        self.chunks = chunks
        # The title is part of a chunk's searchable surface: "commission
        # grid" is in the document name, not in every section body.
        self.docs = [_tokens(f"{c['title']} {c['section']} {c['text']}")
                     for c in chunks]
        self.lens = [len(d) for d in self.docs]
        self.avg_len = (sum(self.lens) / len(self.lens)) if self.lens else 0.0
        self.tf: list[Counter] = [Counter(d) for d in self.docs]
        df: Counter = Counter()
        for d in self.docs:
            df.update(set(d))
        self.df = df
        n = max(len(chunks), 1)
        self.idf = {t: math.log(1 + (n - c + 0.5) / (c + 0.5))
                    for t, c in df.items()}
        # A term the corpus has never seen is the rarest term there is.
        self.max_idf = max(self.idf.values(), default=1.0)
        # L2-normalised tf-idf vectors, so a cosine is a dot product.
        self.vecs: list[dict[str, float]] = []
        for tf in self.tf:
            v = {t: (1 + math.log(f)) * self.idf.get(t, 0.0)
                 for t, f in tf.items()}
            norm = math.sqrt(sum(x * x for x in v.values())) or 1.0
            self.vecs.append({t: x / norm for t, x in v.items()})

    def bm25(self, q: list[str], i: int, k1: float = 1.4,
             b: float = 0.75) -> float:
        tf, dl = self.tf[i], self.lens[i] or 1
        score = 0.0
        for t in q:
            f = tf.get(t, 0)
            if not f:
                continue
            score += self.idf.get(t, 0.0) * (f * (k1 + 1)) / (
                f + k1 * (1 - b + b * dl / (self.avg_len or 1)))
        return score

    def cosine(self, qvec: dict[str, float], i: int) -> float:
        v = self.vecs[i]
        if len(qvec) > len(v):
            qvec, v = v, qvec
        return sum(w * v.get(t, 0.0) for t, w in qvec.items())

    def coverage(self, q: list[str], i: int) -> float:
        """How much of the question's INFORMATION this chunk actually matches.

        A term absent from the whole corpus is maximally informative, not
        uninformative - "does my plan cover a holiday in Lisbon" is mostly
        about Lisbon, and a brochure that matches "plan" and "cover" has
        answered none of it. Scoring absent terms as zero idf is what lets a
        confident answer be composed out of the generic half of a question.
        """
        if not q:
            return 0.0
        tf = self.tf[i]
        total = matched = 0.0
        for t in set(q):
            w = self.idf.get(t, self.max_idf)
            total += w
            if tf.get(t):
                matched += w
        return matched / total if total else 0.0

    def qvec(self, q: list[str]) -> dict[str, float]:
        tf = Counter(q)
        v = {t: (1 + math.log(f)) * self.idf.get(t, 0.0) for t, f in tf.items()}
        norm = math.sqrt(sum(x * x for x in v.values())) or 1.0
        return {t: x / norm for t, x in v.items()}


@lru_cache(maxsize=1)
def _index() -> _Index:
    return _Index(corpus())


def reset_index() -> None:
    _index.cache_clear()


# ---------------------------------------------------------------------------
# Pre-filter (KB-2, KB-6)
# ---------------------------------------------------------------------------
def _in_force(chunk: dict, as_of: date) -> bool:
    # Dating filters only the documents that exist in more than one version.
    # Everything else carries its effective date as provenance for the
    # citation, and stays retrievable.
    if not chunk.get("date_sensitive"):
        return True
    frm, to = chunk.get("effective_from"), chunk.get("effective_to")
    if frm and as_of < date.fromisoformat(str(frm)):
        return False
    if to and as_of > date.fromisoformat(str(to)):
        return False
    return True


def prefilter(lob: str, scopes: list[str], as_of: str | None = None,
              product: str | None = None) -> tuple[list[int], dict[str, int]]:
    """Indices that survive the structural filter, and a count of what did not."""
    idx = _index()
    when = date.fromisoformat(as_of) if as_of else None
    keep: list[int] = []
    dropped = {"lob": 0, "scope": 0, "effective_date": 0, "product": 0}
    for i, c in enumerate(idx.chunks):
        if c["lob"] != lob:
            dropped["lob"] += 1
            continue
        if c["scope"] not in scopes:
            dropped["scope"] += 1
            continue
        if when and not _in_force(c, when):
            dropped["effective_date"] += 1
            continue
        if product and c["product"] not in (None, product):
            dropped["product"] += 1
            continue
        keep.append(i)
    return keep, dropped


# ---------------------------------------------------------------------------
# Rerank
# ---------------------------------------------------------------------------
_DOC_TYPE_HINTS = {
    "claim": ("claim_guide", "process"),
    "claims": ("claim_guide", "process"),
    "cashless": ("claim_guide", "process"),
    "premium": ("rate_table",),
    "rate": ("rate_table",),
    "commission": ("commercial",),
    "payout": ("commercial",),
    "exclusion": ("exclusions_schedule", "policy_wording"),
    "excluded": ("exclusions_schedule", "policy_wording"),
    "waiting": ("policy_wording", "faq"),
    "underwriting": ("underwriting_guide",),
    "objection": ("commercial",),
}

_AUTHORITY_WEIGHT = {"irdai": 0.10, "protec": 0.05, "third_party": 0.0}


def _deterministic_rerank(cands: list[Candidate], query: str) -> None:
    """Features a cross-encoder would learn, written down instead.

    Cheap, explainable and stable across runs, which matters more in a
    regulated demo than the last two points of nDCG.
    """
    q = query.lower()
    qt = set(_tokens(query))
    hinted: set[str] = set()
    for word, types in _DOC_TYPE_HINTS.items():
        if word in q:
            hinted.update(types)

    for cand in cands:
        c = cand.chunk
        bump, why = 0.0, []
        section_hits = len(qt & set(_tokens(c["section"])))
        if section_hits:
            bump += 0.12 * section_hits
            why.append(f"section match x{section_hits}")
        # An exact multi-word phrase is the strongest cheap signal there is.
        for n in (4, 3, 2):
            words = _tokens(query)
            grams = {" ".join(words[i:i + n]) for i in range(len(words) - n + 1)}
            if any(g in c["text"].lower() for g in grams):
                bump += 0.06 * n
                why.append(f"{n}-gram phrase")
                break
        if hinted and c["doc_type"] in hinted:
            bump += 0.10
            why.append(f"doc_type {c['doc_type']}")
        auth = _AUTHORITY_WEIGHT.get(c["authority"], 0.0)
        if auth:
            bump += auth
            why.append(f"authority {c['authority']}")
        if c.get("effective_to") is None:
            bump += 0.02
            why.append("current version")
        cand.rerank = round(bump, 4)
        cand.reasons.extend(why)


def _llm_rerank(cands: list[Candidate], query: str) -> None:
    """The drop-in for a real cross-encoder: a small model scores relevance.

    Off by default - it costs a call per turn and the deterministic reranker
    is good enough for the corpus this demo carries.
    """
    from app.llm.bedrock import score_relevance

    scored = score_relevance(query, [
        {"chunk_id": c.chunk["chunk_id"],
         "text": f"{c.chunk['section']}: {c.chunk['text'][:400]}"}
        for c in cands])
    for cand in cands:
        s = scored.get(cand.chunk["chunk_id"])
        if s is not None:
            cand.rerank = round(0.4 * s, 4)
            cand.reasons.append(f"llm rerank {s:.2f}")


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------
def search(query: str, lob: str, scopes: list[str], k: int | None = None,
           as_of: str | None = None, product: str | None = None
           ) -> tuple[list[Candidate], list[Candidate], dict]:
    """Returns (accepted, rejected, stats).

    `rejected` is every candidate that entered ranking and did not make the
    cut, with its scores - OB-4. Chunks removed by the pre-filter are counted
    in `stats`, not returned: they were never candidates.
    """
    cfg = settings()
    k = k or cfg.retrieval_k
    idx = _index()
    keep, dropped = prefilter(lob, scopes, as_of=as_of, product=product)
    q = _tokens(query)
    stats = {"corpus": len(idx.chunks), "after_prefilter": len(keep),
             "dropped": dropped, "floor": cfg.retrieval_floor,
             "query_terms": q, "candidates": 0, "accepted": 0, "rejected": 0}
    if not q or not keep:
        return [], [], stats

    qvec = idx.qvec(q)
    scored = [(i, idx.bm25(q, i), idx.cosine(qvec, i)) for i in keep]

    # Reciprocal rank fusion: rank-based, so the two scales never need to be
    # calibrated against each other.
    by_bm = sorted(scored, key=lambda r: -r[1])
    by_cos = sorted(scored, key=lambda r: -r[2])
    rank_bm = {i: n for n, (i, *_ ) in enumerate(by_bm)}
    rank_cos = {i: n for n, (i, *_ ) in enumerate(by_cos)}
    kappa = 20.0

    cands: list[Candidate] = []
    for i, bm, cos in scored:
        if bm <= 0 and cos <= 0:
            continue                      # no lexical or vector signal at all
        fused = 1 / (kappa + rank_bm[i] + 1) + 1 / (kappa + rank_cos[i] + 1)
        cands.append(Candidate(chunk=idx.chunks[i], bm25=bm, cosine=cos,
                               fused=fused))
    if not cands:
        return [], [], stats

    cands.sort(key=lambda c: -c.fused)
    cands = cands[:cfg.retrieval_candidates]

    if cfg.rerank_with_llm:
        try:
            _llm_rerank(cands, query)
        except Exception:                                    # noqa: BLE001
            _deterministic_rerank(cands, query)
    else:
        _deterministic_rerank(cands, query)

    # The final score is ABSOLUTE, not relative to the best hit in this
    # query. Normalising to the top candidate would mean the best of five bad
    # chunks always scores 1.0, and the KB-5 floor would never fire - the
    # failure mode where a confident answer is composed from nothing.
    #
    # Metadata modulates relevance, it does not manufacture it: the reranker
    # is a multiplier on a real lexical/semantic match, so a chunk with the
    # right doc_type and no term overlap still scores near zero.
    by_chunk = {id(idx.chunks[i]): i for i in keep}
    for c in cands:
        lexical = min(c.bm25 / 6.0, 1.0)
        base = 0.6 * lexical + 0.4 * c.cosine
        c.coverage = idx.coverage(q, by_chunk[id(c.chunk)])
        c.score = round(base * (1 + c.rerank) * c.coverage, 4)
        if c.coverage < 0.5:
            c.reasons.append(f"covers {c.coverage:.0%} of the question")
    cands.sort(key=lambda c: -c.score)

    accepted = [c for c in cands[:k] if c.score >= cfg.retrieval_floor]
    for c in cands:
        if c not in accepted:
            c.rejected = ("below floor" if c.score < cfg.retrieval_floor
                          else f"outside top {k}")
    rejected = [c for c in cands if c.rejected]
    stats |= {"candidates": len(cands), "accepted": len(accepted),
              "rejected": len(rejected)}
    return accepted, rejected, stats
