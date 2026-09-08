"""Render the whole document corpus from data/products.yaml (KB-7).

    uv run python scripts/build_corpus.py

Writes markdown documents plus a chunks.json carrying the KB-1 metadata to
data/generated/. The runtime works without it - app/knowledge/corpus.py
generates the same chunks in memory - but having the files on disk is what
lets a reviewer open a document and check that the brochure and the wording
say the same number.
"""
from __future__ import annotations

import json
import sys

from app.knowledge.corpus import GENERATED
from app.knowledge.generate import as_markdown, chunks, documents


def main() -> int:
    docs = documents()
    rows = chunks(docs)

    # Clear the FILES, not the directories. OneDrive holds a handle on a
    # synced folder and `rmtree` fails with WinError 5 on the directory even
    # though every file in it was removable - which failed the build for a
    # reason that has nothing to do with the build.
    if GENERATED.exists():
        for f in sorted(GENERATED.rglob("*")):
            if f.is_file():
                try:
                    f.unlink()
                except OSError:
                    pass
    (GENERATED / "documents").mkdir(parents=True, exist_ok=True)

    for d in docs:
        (GENERATED / "documents" / f"{d['doc_id']}.md").write_text(
            as_markdown(d), encoding="utf-8")
        # Bedrock Knowledge Bases read a `<file>.metadata.json` sidecar and
        # make those attributes filterable. Without it KB-2 has nothing to
        # filter on, and an agent-scoped document would be retrievable by a
        # customer bot - the corpus scope would become a suggestion rather
        # than a boundary.
        (GENERATED / "documents" / f"{d['doc_id']}.md.metadata.json").write_text(
            json.dumps({"metadataAttributes": {
                "doc_id": d["doc_id"],
                "lob": d["lob"],
                "scope": d["scope"],
                "authority": d["authority"],
                "doc_type": d["doc_type"],
                "product": d.get("product") or "",
                "version": d.get("version") or "",
                "effective_from": d.get("effective_from") or "",
                "source": d.get("source") or d.get("title") or d["doc_id"],
            }}, indent=1), encoding="utf-8")

    (GENERATED / "chunks.json").write_text(
        json.dumps(rows, indent=1, ensure_ascii=False), encoding="utf-8")

    by_scope: dict[str, int] = {}
    by_lob: dict[str, int] = {}
    for c in rows:
        by_scope[c["scope"]] = by_scope.get(c["scope"], 0) + 1
        by_lob[c["lob"]] = by_lob.get(c["lob"], 0) + 1

    print(f"\n{len(docs)} documents, {len(rows)} chunks -> {GENERATED}")
    print(f"  by lob   : {by_lob}")
    print(f"  by scope : {by_scope}")
    print("\nEvery figure in every document came from data/products.yaml.\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
