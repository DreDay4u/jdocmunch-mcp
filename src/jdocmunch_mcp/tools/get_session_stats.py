"""get_session_stats — concise session view: latency_per_tool + total tokens saved."""

from __future__ import annotations

from typing import Optional

from ..storage.token_tracker import (
    SAVINGS_BASIS,
    budget_status,
    get_session_response_tokens,
    get_total_saved,
    latency_stats,
)


def get_session_stats(storage_path: Optional[str] = None) -> dict:
    """Return latency_per_tool (in-memory ring) + cumulative tokens_saved.

    Lightweight wrapper that an agent can call to self-monitor without
    opting into the SQLite sink. For windowed historical analysis use
    ``analyze_perf(window=...)``.

    v1.104.0: also reports ``session_response_tokens`` (context served this
    session) and, when ``JDOCMUNCH_SESSION_TOKEN_BUDGET`` is set, the
    advisory ``budget`` block ({limit, spent, state}) — suite parity with
    jcodemunch-mcp. Advisory only; nothing is ever blocked or truncated.
    """
    stats = {
        "latency_per_tool": latency_stats(),
        "total_tokens_saved": get_total_saved(storage_path),
        "savings_basis": SAVINGS_BASIS,
        "session_response_tokens": get_session_response_tokens(),
    }
    budget = budget_status()
    if budget is not None:
        stats["budget"] = budget
    return stats
