"""watch: reindex batches run in a child process, so peak RSS returns on exit.

``incremental_save`` rewrites the whole monolith however few files changed,
so every batch's peak is the largest index's full rebuild and the allocator
floor it leaves never fully returns: a watch daemon 9 h old measured 1.34 GB
RSS with a live set near zero. The fix runs each batch in a throwaway
interpreter (``_watch_child``); the parent's own memory no longer depends on
corpus size.

Acceptance: the default path applies a batch's edits through a real child
(``sys.executable -m jdocmunch_mcp._watch_child``) and leaves the parent's
``_INDEX_CACHE`` untouched; ``JDOCMUNCH_WATCH_INPROCESS=1`` restores the
in-process thread path; a child failure surfaces through the existing
"reindex failed" warning; and the shape-pattern lookup reads the summary
sidecar instead of parsing the monolith, falling back to the full load when
no summary exists.
"""
import asyncio
import io
import json
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _default_isolation(monkeypatch):
    """Deterministic default: no ambient JDOCMUNCH_WATCH_INPROCESS."""
    monkeypatch.delenv("JDOCMUNCH_WATCH_INPROCESS", raising=False)


def _fixture(tmp_path):
    fx = tmp_path / "repro"
    fx.mkdir()
    (fx / "README.md").write_text(
        "# Visible document\n\nInitial body.\n", encoding="utf-8")
    (fx / "IGNORED.md").write_text("# Ignored\n\nInitial.\n", encoding="utf-8")
    store = tmp_path / "store"
    store.mkdir()
    return fx, store


def _index(fx, store, **kw):
    from jdocmunch_mcp.tools.index_local import index_local
    return index_local(path=str(fx), name="repro", use_embeddings=False,
                       use_ai_summaries=False, storage_path=str(store), **kw)


def _snapshot(store):
    from jdocmunch_mcp.storage.doc_store import DocStore
    i = DocStore(base_path=str(store)).load_index("local", "repro")
    return (sorted(i.doc_paths),
            sorted(s["content_hash"] for s in i.sections),
            dict(i.file_hashes))


def _fire(paths, fx, store):
    from jdocmunch_mcp import watch as W
    roots = {str(Path(fx).resolve()): "repro"}
    asyncio.run(W._handle_changes(
        [(1, str(Path(p).resolve())) for p in paths],
        roots, str(store), False, True, None))


def _cache_entries():
    from jdocmunch_mcp.storage import doc_store as ds
    return len(ds._INDEX_CACHE)


class TestChildProcessDefault:
    def test_a_batch_applies_its_edits_through_a_real_child(self, tmp_path):
        fx, store = _fixture(tmp_path)
        _index(fx, store)
        _snapshot(store)  # warm nothing: reads happen in THIS process though

        (fx / "README.md").write_text(
            "# Visible document\n\nEdited via the child.\n", encoding="utf-8")
        (fx / "NEW.md").write_text("# New doc\n\nAdded.\n", encoding="utf-8")
        before_cache = _cache_entries()
        _fire([fx / "README.md", fx / "NEW.md"], fx, store)

        docs, _hashes, files = _snapshot(store)
        assert "NEW.md" in docs, "the child must have indexed the new file"

    def test_the_parent_cache_stays_empty_across_batches(self, tmp_path):
        """The point of the child: the parent never parses the monolith."""
        from jdocmunch_mcp.storage import doc_store as ds
        fx, store = _fixture(tmp_path)
        _index(fx, store)
        ds._INDEX_CACHE.clear()
        try:
            for n in range(3):
                (fx / "README.md").write_text(
                    f"# Visible document\n\nEdit {n}.\n", encoding="utf-8")
                _fire([fx / "README.md"], fx, store)
                assert len(ds._INDEX_CACHE) == 0, (
                    "the watch parent must not retain a parsed index"
                )
        finally:
            ds._INDEX_CACHE.clear()

    def test_inprocess_env_restores_the_thread_path(self, tmp_path, monkeypatch):
        fx, store = _fixture(tmp_path)
        _index(fx, store)

        calls = []
        import jdocmunch_mcp.tools.index_local as il
        real = il.index_local

        def spy(**kwargs):
            calls.append(kwargs)
            return real(**kwargs)

        monkeypatch.setattr(il, "index_local", spy)
        monkeypatch.setenv("JDOCMUNCH_WATCH_INPROCESS", "1")

        (fx / "README.md").write_text("# Edited\n\nIn-process.\n", encoding="utf-8")
        _fire([fx / "README.md"], fx, store)

        assert len(calls) == 1, "JDOCMUNCH_WATCH_INPROCESS=1 must take the thread path"
        assert calls[0]["incremental"] is True
        assert calls[0]["paths"] == [str((fx / "README.md").resolve())]

    def test_a_child_failure_surfaces_as_the_reindex_warning(self, tmp_path, caplog):
        fx, store = _fixture(tmp_path)
        _index(fx, store)
        # A repo name DocStore._safe_repo_component rejects: the child must
        # exit nonzero and the parent must log, not raise.
        roots = {str(fx.resolve()): "bad/name"}

        async def scenario():
            from jdocmunch_mcp import watch as W
            await W._handle_changes(
                [(1, str((fx / "README.md").resolve()))],
                roots, str(store), False, True, None)

        with caplog.at_level("WARNING"):
            asyncio.run(scenario())  # must not raise

        assert any("reindex failed" in r.message for r in caplog.records)

    def test_reindex_in_subprocess_raises_with_the_child_stderr(self, tmp_path):
        from jdocmunch_mcp import watch as W

        fx, _store = _fixture(tmp_path)

        async def scenario():
            await W._reindex_in_subprocess(
                str(fx), "bad/name", [str(fx / "README.md")], str(tmp_path), False)

        with pytest.raises(RuntimeError, match="_watch_child exited"):
            asyncio.run(scenario())


class TestWatchChildModule:
    def test_direct_run_applies_paths(self, tmp_path, monkeypatch):
        from jdocmunch_mcp import _watch_child

        fx, store = _fixture(tmp_path)
        _index(fx, store)
        before = _snapshot(store)

        (fx / "README.md").write_text("# Edited\n\nBy hand.\n", encoding="utf-8")
        monkeypatch.setattr(
            "sys.argv", ["_watch_child", str(fx), "repro", str(store), "0"])
        monkeypatch.setattr(
            "sys.stdin", io.StringIO(str(fx / "README.md") + "\n"))
        assert _watch_child.main() == 0

        after = _snapshot(store)
        assert after[0] == before[0], "no doc added or dropped"
        assert after[1] != before[1], "the edited file's sections must be re-hashed"

    def test_bad_argc_prints_usage_and_exits_2(self, capsys):
        from jdocmunch_mcp import _watch_child

        assert _watch_child.main(["only-one-arg"]) == 2
        assert "usage:" in capsys.readouterr().err

    def test_empty_stdin_is_a_clean_no_op(self, tmp_path, monkeypatch):
        from jdocmunch_mcp import _watch_child

        monkeypatch.setattr(
            "sys.argv", ["_watch_child", str(tmp_path), "repro", str(tmp_path), "0"])
        monkeypatch.setattr("sys.stdin", io.StringIO("\n \n"))
        assert _watch_child.main() == 0


class TestSummaryFirstShapePatterns:
    def test_patterns_come_from_the_summary_without_parsing_the_monolith(
        self, tmp_path,
    ):
        from jdocmunch_mcp.storage import doc_store as ds
        from jdocmunch_mcp import watch as W

        fx, store = _fixture(tmp_path)
        _index(fx, store, extra_ignore_patterns=["IGNORED.md"])
        ds._INDEX_CACHE.clear()
        try:
            patterns = W._stored_shape_patterns(str(store), "repro")
            assert patterns == ["IGNORED.md"]
            assert len(ds._INDEX_CACHE) == 0, (
                "the summary read must not parse (and cache) the monolith"
            )
        finally:
            ds._INDEX_CACHE.clear()

    def test_missing_summary_falls_back_to_the_full_load(self, tmp_path):
        from jdocmunch_mcp.storage import doc_store as ds
        from jdocmunch_mcp import watch as W

        fx, store = _fixture(tmp_path)
        _index(fx, store, extra_ignore_patterns=["IGNORED.md"])
        summary = store / "local" / "repro.summary.json"
        assert summary.exists()
        summary.unlink()

        ds._INDEX_CACHE.clear()
        try:
            patterns = W._stored_shape_patterns(str(store), "repro")
            assert patterns is None  # no summary on disk
            # The corpus-filter caller falls back to the monolith:
            gitignore_spec, extra_spec = W._corpus_filters(
                str(fx), "repro", str(store))
            assert extra_spec is not None, "fallback must still apply the patterns"
            assert extra_spec.match_file("IGNORED.md")
            assert len(ds._INDEX_CACHE) >= 1, "the fallback used the full load"
        finally:
            ds._INDEX_CACHE.clear()

    def test_corpus_filters_apply_stored_patterns_via_the_summary(self, tmp_path):
        from jdocmunch_mcp.storage import doc_store as ds
        from jdocmunch_mcp import watch as W

        fx, store = _fixture(tmp_path)
        _index(fx, store, extra_ignore_patterns=["IGNORED.md"])
        # Discovery dropped IGNORED.md from the corpus; editing it through the
        # watcher must not re-admit it (jdoc#116 behavior, summary-sourced).
        (fx / "IGNORED.md").write_text("# Ignored\n\nEdited.\n", encoding="utf-8")
        _snapshot(store)  # baseline with cache warm state irrelevant
        ds._INDEX_CACHE.clear()
        try:
            before = _snapshot(store)
            _fire([fx / "IGNORED.md"], fx, store)
            assert _snapshot(store) == before, "watcher re-admitted an excluded file"
        finally:
            ds._INDEX_CACHE.clear()
