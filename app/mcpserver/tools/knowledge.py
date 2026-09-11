"""Knowledge / RAG tools.

Retrieval is metadata-pre-filtered by lob and corpus scope BEFORE ranking, so
entitlement is structural rather than a prompt instruction (KB-2). The tool
returns the accepted chunks to the model and puts everything it rejected, with
scores, under `_trace`.

**Corpus scope is read from `runtime.context`, never modelled.** It comes from
the bound agent's `corpus_scope`; a scope the model could pass is a scope the
model could choose, and a customer bot whose model asked for the "agent" scope
would be handed the commission grid by a pre-filter doing as it was told.
Entitlement cannot be an argument.

An empty result is a correct answer: below the score floor the tool returns
nothing and says so, and the model is told (by the prompt) to offer a human
rather than answer from general knowledge (KB-5).
"""
from __future__ import annotations

from typing import Annotated, Optional

from langchain.tools import tool, ToolRuntime

from app.knowledge.retriever import search as _local_search


def _search(query, lob, scopes, k=None, as_of=None, product=None):
    """Bedrock Knowledge Base when one is deployed, the local index when not.
    Both return the same `(accepted, rejected, stats)`."""
    from app.knowledge import kb

    if kb.configured():
        return kb.search(query, lob, scopes, k=k, as_of=as_of, product=product)
    return _local_search(query, lob, scopes, k=k, as_of=as_of, product=product)


def _run(query: str, lob: str, scopes: list[str] | None, k: int | None,
         as_of: str | None) -> dict:
    accepted, rejected, stats = _search(query, lob, scopes or ["public"],
                                        k=k, as_of=as_of)
    # NUMBERED refs, and that is the whole point. Handed `PHS-WORDING-V2#1` the
    # model paraphrases it - underscore and hash gone - and a well-sourced
    # answer is refused on a failed string compare. `[1]` has no fuzzy form:
    # resolving a citation becomes an array index, not a matching problem. The
    # real chunk_id travels alongside for the trace, so nothing is lost.
    chunks = []
    for n, c in enumerate(accepted, 1):
        d = c.as_dict()
        d["ref"] = n
        chunks.append(d)

    return {
        "chunks": chunks,
        "grounded": bool(accepted),
        "scopes_searched": scopes or ["public"],
        "instruction": (
            "Each passage has a `ref` number. Cite it in square brackets - "
            "[1], [2] - after every sentence you take from that passage. Use "
            "the number only, never the chunk_id. If this list is empty, say "
            "you do not have an approved source and offer a colleague - do not "
            "answer from general knowledge."),
        "_trace": {
            "rejected": [c.as_dict(with_text=False) for c in rejected],
            "stats": stats,
        },
    }


def _log_and_strip(result: dict, query: str, as_of: str | None, runtime) -> dict:
    """Pull the rejected candidates and stats out of the tool result into the
    trace (OB-4), and hand the model only the accepted chunks. The `_trace`
    block is internal - a customer's answer must be grounded in what was
    accepted, and the model should never see what was thrown away."""
    meta = result.pop("_trace", {}) or {}
    trace = getattr(getattr(runtime, "context", None), "trace", None)
    if trace is not None:
        stats = meta.get("stats", {})
        chunks = result.get("chunks", [])
        trace.add("retrieve", (query or "")[:120],
                  accepted=len(chunks), rejected=len(meta.get("rejected", [])),
                  corpus=stats.get("corpus"),
                  after_prefilter=stats.get("after_prefilter"),
                  dropped_by_prefilter=stats.get("dropped"),
                  floor=stats.get("floor"), as_of=as_of,
                  accepted_chunks=[{k: c.get(k) for k in
                                    ("chunk_id", "score", "source", "section",
                                     "page", "authority", "version")}
                                   for c in chunks],
                  rejected_chunks=meta.get("rejected", []))
    return result


@tool(extras={"tags": {"lob": "health", "persona": "customer|agent"},
              "authority": "none"})
def kb_search_health(
    query: str,
    as_of: Annotated[Optional[str], "Policy start date (YYYY-MM-DD) when the "
                     "question is about a policy already held; wordings are "
                     "effective-dated."] = None,
    runtime: ToolRuntime = None,
) -> dict:
    """Search approved health product, process and regulatory content. Returns
    passages with source, section and page for citation."""
    scopes = getattr(runtime.context, "corpus_scope", None) if runtime else None
    return _log_and_strip(_run(query, "health", scopes, None, as_of),
                          query, as_of, runtime)


@tool(extras={"tags": {"lob": "motor", "persona": "customer|agent"},
              "authority": "none"})
def kb_search_motor(
    query: str,
    as_of: Annotated[Optional[str], "Policy start date (YYYY-MM-DD) when the "
                     "question is about a policy already held."] = None,
    runtime: ToolRuntime = None,
) -> dict:
    """Search approved motor product, process and regulatory content. Returns
    passages with source, section and page for citation."""
    scopes = getattr(runtime.context, "corpus_scope", None) if runtime else None
    return _log_and_strip(_run(query, "motor", scopes, None, as_of),
                          query, as_of, runtime)
