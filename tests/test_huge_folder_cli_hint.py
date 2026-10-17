"""A slow or capped index_local call names the CLI command for that folder.

An MCP client times a tool call out, and a folder of tens of thousands of
files can take minutes to index. `jdocmunch-mcp index-local` runs the same
code with no timeout, so the response says so where the caller can act.
"""

import pytest

from jdocmunch_mcp.tools import index_local as il_mod
from jdocmunch_mcp.tools.index_local import index_local


@pytest.fixture(autouse=True)
def _no_savings_share(monkeypatch):
    monkeypatch.setenv("JDOCMUNCH_SHARE_SAVINGS", "0")


def _docs(root, count, start=0):
    root.mkdir(exist_ok=True)
    for i in range(start, start + count):
        (root / f"doc{i}.md").write_text(
            f"# Doc {i}\n\nBody of document {i}.\n", encoding="utf-8"
        )
    return root


def _run(folder, storage, **kw):
    return index_local(
        path=str(folder),
        use_ai_summaries=False,
        use_embeddings=False,
        storage_path=str(storage),
        **kw,
    )


def _cli_warnings(result):
    return [w for w in result.get("warnings", []) if "jdocmunch-mcp index-local" in w]


def test_full_path_cap_warning_names_the_cli(tmp_path):
    folder = _docs(tmp_path / "docs", 3)
    result = _run(folder, tmp_path / "store", max_files=2, incremental=False)
    assert result["success"] is True
    assert result["truncated"] is True
    cap = [w for w in result["warnings"] if w.startswith("max_files cap hit")]
    assert len(cap) == 1
    assert f'jdocmunch-mcp index-local --path "{folder.resolve()}"' in cap[0]
    assert "keeps the 10,000-file cap" in cap[0]
    # Existing keys keep their meaning.
    assert result["discovered"] == 3
    assert result["indexed"] == 2


def test_incremental_path_cap_warning_names_the_cli(tmp_path):
    folder = _docs(tmp_path / "docs", 2)
    store = tmp_path / "store"
    first = _run(folder, store, max_files=10)
    assert first["success"] is True and first["truncated"] is False
    _docs(folder, 2, start=2)
    second = _run(folder, store, max_files=3)
    assert second["success"] is True
    assert second.get("incremental") is True
    assert second["truncated"] is True
    cap = [w for w in second["warnings"] if w.startswith("max_files cap hit")]
    assert len(cap) == 1
    assert "jdocmunch-mcp index-local --path" in cap[0]


def test_cap_warning_carries_an_explicit_name(tmp_path):
    folder = _docs(tmp_path / "docs", 3)
    result = _run(folder, tmp_path / "store", name="mydocs", max_files=2)
    cap = [w for w in result["warnings"] if w.startswith("max_files cap hit")]
    assert '--name "mydocs"' in cap[0]


def test_long_run_hint_on_full_path(tmp_path, monkeypatch):
    monkeypatch.setattr(il_mod, "LONG_RUN_HINT_SECONDS", -1)
    folder = _docs(tmp_path / "docs", 2)
    result = _run(folder, tmp_path / "store", incremental=False)
    assert result["success"] is True
    hints = [w for w in _cli_warnings(result) if w.startswith("This call took")]
    assert len(hints) == 1
    assert f'--path "{folder.resolve()}"' in hints[0]
    assert "--name" not in hints[0]


def test_long_run_hint_on_incremental_path(tmp_path, monkeypatch):
    folder = _docs(tmp_path / "docs", 2)
    store = tmp_path / "store"
    assert _run(folder, store)["success"] is True
    monkeypatch.setattr(il_mod, "LONG_RUN_HINT_SECONDS", -1)
    _docs(folder, 1, start=2)
    result = _run(folder, store)
    assert result["success"] is True
    assert result.get("incremental") is True
    assert any(w.startswith("This call took") for w in _cli_warnings(result))


def test_no_hint_under_the_threshold(tmp_path):
    folder = _docs(tmp_path / "docs", 2)
    store = tmp_path / "store"
    full = _run(folder, store)
    assert full["success"] is True
    assert _cli_warnings(full) == []
    _docs(folder, 1, start=2)
    inc = _run(folder, store)
    assert inc["success"] is True
    assert _cli_warnings(inc) == []


def test_threshold_default_is_thirty_seconds():
    assert il_mod.LONG_RUN_HINT_SECONDS == 30
