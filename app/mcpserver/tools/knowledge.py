"""Knowledge / RAG tools.

Retrieval is metadata-pre-filtered by lob and corpus scope BEFORE ranking, so
entitlement is structural rather than a prompt instruction (KB-2). The tool
returns the accepted chunks to the model and puts everything it rejected,
with scores, under `_trace` - which the agent strips before the model sees it
and writes into the glass box instead (OB-4).

**`_scopes` is injected, never modelled.** It comes from the bound agent's
`corpus_scope` and is not in the schema the model sees. That is not tidiness:
a scope the model can pass is a scope the model can *choose*, and a customer
bot whose model asks for `scopes=["agent"]` would be handed the commission
grid by a pre-filter doing exactly what it was told. Entitlement cannot be an
argument.

An empty result is a correct answer. Below the score floor the tool returns
nothing and says so, and the agent refuses and offers a human rather than
composing something confident out of four bad chunks (KB-5).
"""
from __future__ import annotations

from app.knowledge.retriever import search as _local_search


def search(query, lob, scopes, k=None, as_of=None, product=None):
    """Bedrock Knowledge Base when one is deployed, the local index when not.

    Chosen here rather than inside the retriever so the local index stays a
    self-contained thing the offline tests exercise directly. Both return the
    same `(accepted, rejected, stats)`, and `stats["backend"]` says which
    answered - the glass box should never be vague about where a citation
    came from.
    """
    from app.knowledge import kb

    if kb.configured():
        return kb.search(query, lob, scopes, k=k, as_of=as_of, product=product)
    return _local_search(query, lob, scopes, k=k, as_of=as_of, product=product)
from app.mcpserver.registry import tool


def _run(query: str, lob: str, scopes: list[str] | None, k: int | None,
         as_of: str | None) -> dict:
    accepted, rejected, stats = search(query, lob, scopes or ["public"],
                                       k=k, as_of=as_of)
    return {
        "chunks": [c.as_dict() for c in accepted],
        "grounded": bool(accepted),
        "scopes_searched": scopes or ["public"],
        "instruction": (
            "Cite a chunk_id in square brackets after every factual sentence "
            "you take from these. If this list is empty, say you do not have "
            "an approved source and offer a colleague - do not answer from "
            "general knowledge."),
        "_trace": {
            "rejected": [c.as_dict(with_text=False) for c in rejected],
            "stats": stats,
        },
    }


@tool(tags={"lob": "health", "persona": "customer|agent"}, authority="none")
def kb_search_health(query: str, as_of: str | None = None,
                     _scopes: list[str] | None = None) -> dict:
    """Search approved HEALTH product, process and regulatory content.

    Returns chunks with source, section and page for citation. Pass `as_of`
    as the policy start date when the question is about a policy the customer
    already holds - wordings are effective-dated and the answer must come
    from the version in force on that date."""
    return _run(query, "health", _scopes, None, as_of)


@tool(tags={"lob": "motor", "persona": "customer|agent"}, authority="none")
def kb_search_motor(query: str, as_of: str | None = None,
                    _scopes: list[str] | None = None) -> dict:
    """Search approved MOTOR product, process and regulatory content.

    Returns chunks with source, section and page for citation."""
    return _run(query, "motor", _scopes, None, as_of)
