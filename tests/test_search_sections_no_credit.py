"""A search earns no savings credit; opening a section keeps its credit.

Maintainer's ruling, 2026-10-17 (the suite-wide L-180): `search_sections`
credited the whole size of every document holding a returned section. That is
work a text search would not have done either, and the same document was
credited again when the agent opened a section. The search now reports zero
and adds nothing to the running or shared total. Every savings figure carries
`savings_basis: "modeled"`, because the 1.x keys can't be renamed.
"""

from __future__ import annotations

import pytest

from jdocmunch_mcp.storage import token_tracker
from jdocmunch_mcp.storage.token_tracker import SAVINGS_BASIS, get_total_saved

_BODY = "To configure the widget you must set the `widget_timeout` option. " * 40


@pytest.fixture
def repo(tmp_path, monkeypatch):
    monkeypatch.setenv("JDOCMUNCH_SHARE_SAVINGS", "0")
    from jdocmunch_mcp.tools.index_local import index_local

    src = tmp_path / "docs"
    src.mkdir()
    for i in range(4):
        (src / f"doc{i}.md").write_text(
            f"# Doc {i}\n\n## Configuring the widget\n\n{_BODY}\n\n## Other {i}\n\nmore\n",
            encoding="utf-8",
        )
    storage = str(tmp_path / "store")
    r = index_local(path=str(src), name="nocredit", use_ai_summaries=False,
                    use_embeddings=False, storage_path=storage)
    assert r.get("success"), r
    return r["repo"], storage


def test_search_reports_zero_and_leaves_the_total_alone(repo, monkeypatch):
    from jdocmunch_mcp.tools.search_sections import search_sections
    name, storage = repo
    shared = []
    monkeypatch.setattr(token_tracker, "_share_savings", lambda *a, **k: shared.append(a))
    before = get_total_saved(storage)

    out = search_sections(repo=name, query="configure widget timeout", storage_path=storage)

    assert out["results"]
    meta = out["_meta"]
    assert meta["tokens_saved"] == 0
    assert all(v == 0 for v in meta["cost_avoided"].values())
    assert get_total_saved(storage) == before
    assert meta["savings_basis"] == SAVINGS_BASIS == "modeled"
    assert shared == []


def test_opening_a_section_keeps_its_credit(repo):
    from jdocmunch_mcp.tools.get_section import get_section
    from jdocmunch_mcp.tools.search_sections import search_sections
    name, storage = repo
    hit = search_sections(repo=name, query="configure widget timeout",
                          storage_path=storage)["results"][0]
    before = get_total_saved(storage)

    out = get_section(repo=name, section_id=hit["id"], storage_path=storage)

    assert out["_meta"]["tokens_saved"] > 0
    assert out["_meta"]["savings_basis"] == "modeled"
    assert get_total_saved(storage) == before + out["_meta"]["tokens_saved"]


def test_session_stats_names_the_basis(repo):
    from jdocmunch_mcp.tools.get_session_stats import get_session_stats
    _name, storage = repo
    assert get_session_stats(storage_path=storage)["savings_basis"] == "modeled"
