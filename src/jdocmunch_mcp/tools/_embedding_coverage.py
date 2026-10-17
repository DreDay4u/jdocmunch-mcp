"""Embedding coverage disclosure for the indexing tools (jdoc#107).

#107's reporter lost a 5,316-vector sidecar three times and only found out by
writing their own external check — comparing sidecar rows against
`section_count` after every reindex and rebuilding below 50%. The index tools
emitted nothing: exit 0, no warning, and `search_sections` kept answering with
a semantic channel that ranked almost nothing.

⚠⚠ The write bug is fixed at its source; this exists because the SILENCE was
the other half of the defect. A coverage collapse from any future cause — a
provider timing out mid-batch, a half-written sidecar, an identity rotation —
is now visible in the response instead of needing a user to instrument us.

Counted off the sidecar's KEYS, never its vectors: a 26k-section corpus costs
no memory here.
"""

from __future__ import annotations

from typing import Optional

# Below this, the semantic channel is degraded enough that the caller should
# be told in words, not just handed a ratio. Matches the threshold #107's
# reporter chose independently for their external rebuild trigger.
COVERAGE_WARN_BELOW = 0.5


def attach_provider_failure(result: dict, warnings: Optional[list] = None) -> dict:
    """Say so when `semantic_search` is true but the provider produced nothing.

    `semantic_search` means "a provider is configured", and on 1.x that key
    keeps its meaning. When the provider failed to load or its last embed pass
    raised, ``embedding_provider_error`` carries the reason beside it, and a
    warning says search is lexical. Absent when nothing failed.
    """
    if not result.get("semantic_search"):
        return result
    try:
        from ..embeddings.provider import provider_error

        reason = provider_error()
    except Exception:
        return result
    if not reason:
        return result
    result["embedding_provider_error"] = reason
    if warnings is not None:
        try:
            from ..embeddings.provider import get_provider_name
            offline = get_provider_name() == "fastembed"
        except Exception:
            offline = False
        repair = (
            " To repair a missing or damaged FastEmbed model, run "
            "`jdocmunch-mcp init --with-embeddings` (it downloads the model "
            "once, with your consent)."
            if offline else
            " The reason above names what failed, such as a missing model "
            "file or an API error."
        )
        warnings.append(
            "Semantic search is configured but produced no vectors this run "
            f"({reason}). Search is lexical (BM25) only until the provider "
            "loads." + repair
        )
    return result


def attach_embedding_coverage(
    result: dict,
    *,
    storage_path: Optional[str],
    owner: str,
    name: str,
    index,
    warnings: Optional[list] = None,
) -> dict:
    """Add ``embedded_sections`` / ``embedding_coverage`` to a tool response.

    No-op when the index has no sections or no sidecar exists (a lexical-only
    index must not grow a 0.0 coverage field that reads like a regression).
    """
    attach_provider_failure(result, warnings)
    if index is None:
        return result
    try:
        from ..embeddings import cache as _emb_cache

        stored = _emb_cache.stored_hashes(storage_path, owner, name)
        if not stored:
            return result
        sections = getattr(index, "sections", None) or []
        total = len(sections)
        if not total:
            return result
        covered = sum(1 for s in sections if s.get("content_hash") in stored)
        result["embedded_sections"] = covered
        result["embedding_coverage"] = round(covered / total, 4)
        if covered / total < COVERAGE_WARN_BELOW and warnings is not None:
            warnings.append(
                f"Embedding coverage is {covered}/{total} sections "
                f"({covered / total:.1%}). Semantic search will rank most of "
                f"this corpus at zero. Re-index with incremental=false to "
                f"rebuild the embeddings sidecar."
            )
    except Exception:
        pass  # disclosure is best-effort; never fail an index over it
    return result
