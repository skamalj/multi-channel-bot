"""The corpus the retriever indexes.

Prefers the generated files on disk (`scripts/build_corpus.py` writes ~80
documents to `data/generated/`), and falls back to generating the same
chunks in memory. Both paths come from `products.yaml`, so a checkout with
no generated directory retrieves exactly the same content - which is what
makes the tests runnable on a clean clone.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

GENERATED = Path(__file__).resolve().parents[2] / "data" / "generated"
CHUNKS_FILE = GENERATED / "chunks.json"


@lru_cache(maxsize=1)
def corpus() -> list[dict]:
    if CHUNKS_FILE.exists():
        return json.loads(CHUNKS_FILE.read_text(encoding="utf-8"))
    from app.knowledge.generate import chunks

    return chunks()


def reset() -> None:
    corpus.cache_clear()
    from app.knowledge.retriever import reset_index

    reset_index()


def stats() -> dict:
    rows = corpus()
    docs = {c["doc_id"] for c in rows}
    return {
        "source": "generated files" if CHUNKS_FILE.exists() else "in-memory",
        "documents": len(docs),
        "chunks": len(rows),
        "by_lob": {lob: sum(1 for c in rows if c["lob"] == lob)
                   for lob in sorted({c["lob"] for c in rows})},
        "by_scope": {s: sum(1 for c in rows if c["scope"] == s)
                     for s in sorted({c["scope"] for c in rows})},
        "by_doc_type": {t: sum(1 for c in rows if c["doc_type"] == t)
                        for t in sorted({c["doc_type"] for c in rows})},
    }
