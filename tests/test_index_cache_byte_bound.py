"""_INDEX_CACHE: the byte bound, not just the entry count, must cap retention.

The 8-entry count bound alone let a long-lived serve process retain gigabytes:
a parsed DocIndex holds ~2.5-2.9x its monolith file size as live Python
objects (measured on 25.8k- and 90.3k-section indexes: 19.2 MB file -> 54 MB
retained, 80.3 MB -> 203 MB), and eight such entries is well over a gigabyte.

Acceptance: with a byte budget set, an insert that crosses it evicts LRU
entries until the estimated total fits — but the entry just inserted is never
evicted by its own size; a budget <= 0 restores count-only bounding; the
count bound still applies alongside it; a real load estimates stat-size x
multiplier; and _evict_index_cache keeps the running total honest.
"""
import os

import pytest


@pytest.fixture()
def cache_sandbox(monkeypatch):
    """Isolated cache state with a deterministic size estimator."""
    from jdocmunch_mcp.storage import doc_store as ds

    monkeypatch.setattr(ds, "_INDEX_CACHE", type(ds._INDEX_CACHE)())
    monkeypatch.setattr(ds, "_INDEX_CACHE_ESTIMATES", type(ds._INDEX_CACHE_ESTIMATES)())
    monkeypatch.setattr(ds, "_INDEX_CACHE_BYTES", 0)
    # Budget tests drive the numbers themselves; disable the env override so
    # the machine's environment cannot leak a budget into count-only tests.
    monkeypatch.delenv("JDOCMUNCH_INDEX_CACHE_MAX_BYTES", raising=False)
    sizes = {}

    def fake_estimate(key):
        return sizes.get(key[0], 0)

    monkeypatch.setattr(ds, "_estimate_index_bytes", fake_estimate)
    return ds, sizes


def _budget(ds, monkeypatch, value):
    monkeypatch.setenv("JDOCMUNCH_INDEX_CACHE_MAX_BYTES", str(value))
    assert ds._index_cache_max_bytes() == value


def _put(ds, name, mtime=1):
    key = (f"/idx/{name}.json", mtime)
    ds._index_cache_put(key, object())
    return key


class TestByteBound:
    def test_insert_over_budget_evicts_lru_until_it_fits(self, cache_sandbox, monkeypatch):
        ds, sizes = cache_sandbox
        sizes["/idx/a.json"] = 100
        sizes["/idx/b.json"] = 100
        _budget(ds, monkeypatch, 150)

        a = _put(ds, "a")
        b = _put(ds, "b")

        assert list(ds._INDEX_CACHE) == [b], "the over-budget insert must evict the LRU entry"
        assert ds._INDEX_CACHE_BYTES == 100

    def test_eviction_is_lru_not_fifo_scan(self, cache_sandbox, monkeypatch):
        ds, sizes = cache_sandbox
        for n in "abc":
            sizes[f"/idx/{n}.json"] = 60
        _budget(ds, monkeypatch, 150)

        a = _put(ds, "a")
        b = _put(ds, "b")
        ds._index_cache_get(a)  # a becomes most-recently-used
        c = _put(ds, "c")  # total 180 > 150: b (the true LRU) must go

        assert list(ds._INDEX_CACHE) == [a, c]
        assert ds._INDEX_CACHE_BYTES == 120

    def test_a_single_oversize_entry_is_never_evicted_by_itself(self, cache_sandbox, monkeypatch):
        ds, sizes = cache_sandbox
        sizes["/idx/huge.json"] = 10_000
        _budget(ds, monkeypatch, 10)

        huge = _put(ds, "huge")

        assert list(ds._INDEX_CACHE) == [huge], (
            "one index larger than the whole budget must stay usable, not thrash"
        )
        assert ds._INDEX_CACHE_BYTES == 10_000

    def test_budget_zero_or_negative_disables_the_byte_bound(self, cache_sandbox, monkeypatch):
        ds, sizes = cache_sandbox
        for n in "abcdefghij":
            sizes[f"/idx/{n}.json"] = 1_000_000
        _budget(ds, monkeypatch, 0)

        for n in "abcdefghij":
            _put(ds, n)

        assert len(ds._INDEX_CACHE) == ds._INDEX_CACHE_MAXSIZE == 8

    def test_count_bound_still_applies_alongside_a_large_budget(self, cache_sandbox, monkeypatch):
        ds, sizes = cache_sandbox
        for n in "abcdefghij":
            sizes[f"/idx/{n}.json"] = 1
        _budget(ds, monkeypatch, 10**9)

        keys = [_put(ds, n) for n in "abcdefghij"]

        assert list(ds._INDEX_CACHE) == keys[-8:]
        assert ds._INDEX_CACHE_BYTES == 8

    def test_reput_under_the_same_key_replaces_its_estimate(self, cache_sandbox, monkeypatch):
        ds, sizes = cache_sandbox
        sizes["/idx/a.json"] = 100
        _budget(ds, monkeypatch, 10**9)

        key = _put(ds, "a", mtime=7)
        ds._index_cache_get(key)
        sizes["/idx/a.json"] = 30
        ds._index_cache_put((f"/idx/a.json", 7), object())

        assert ds._INDEX_CACHE_BYTES == 30, "a re-put must not double-count the old estimate"
        assert len(ds._INDEX_CACHE) == 1

    def test_env_budget_is_read_per_insert_not_at_import(self, cache_sandbox, monkeypatch):
        ds, sizes = cache_sandbox
        for n in "ab":
            sizes[f"/idx/{n}.json"] = 100
        _put(ds, "a")
        assert len(ds._INDEX_CACHE) == 1

        _budget(ds, monkeypatch, 150)
        _put(ds, "b")
        assert list(ds._INDEX_CACHE) == [(f"/idx/b.json", 1)], (
            "a budget set after import must take effect on the next insert"
        )

    def test_evict_index_cache_reconciles_the_running_total(self, cache_sandbox, tmp_path):
        ds, sizes = cache_sandbox
        a_path = tmp_path / "a.json"
        b_path = tmp_path / "b.json"
        sizes[str(a_path)] = 40
        sizes[str(b_path)] = 60

        a_key, b_key = (str(a_path), 1), (str(b_path), 1)
        ds._index_cache_put(a_key, object())
        ds._index_cache_put(b_key, object())
        assert ds._INDEX_CACHE_BYTES == 100

        ds._evict_index_cache(a_path)

        assert list(ds._INDEX_CACHE) == [b_key]
        assert ds._INDEX_CACHE_BYTES == 60
        assert set(ds._INDEX_CACHE_ESTIMATES) == set(ds._INDEX_CACHE)


class TestRealLoads:
    def test_a_real_load_estimates_stat_size_times_multiplier(self, tmp_path):
        from jdocmunch_mcp.storage import doc_store as ds
        from jdocmunch_mcp.tools.index_local import index_local

        root = tmp_path / "docs"
        root.mkdir()
        (root / "README.md").write_text("# T\n\n" + ("body\n" * 500), encoding="utf-8")
        store = tmp_path / "store"
        index_local(path=str(root), name="docs", use_embeddings=False,
                    use_ai_summaries=False, storage_path=str(store))

        ds._INDEX_CACHE.clear()
        ds._INDEX_CACHE_ESTIMATES.clear()
        ds._INDEX_CACHE_BYTES = 0
        try:
            index = ds.DocStore(base_path=str(store)).load_index("local", "docs")
            assert index is not None
            (key, _est), = list(ds._INDEX_CACHE_ESTIMATES.items())
            monolith = store / "local" / "docs.json"
            assert key[0] == str(monolith)
            assert ds._INDEX_CACHE_ESTIMATES[key] == (
                os.stat(monolith).st_size * ds._INDEX_CACHE_SIZE_MULTIPLIER
            )
            assert ds._INDEX_CACHE_BYTES == ds._INDEX_CACHE_ESTIMATES[key]
        finally:
            ds._INDEX_CACHE.clear()
            ds._INDEX_CACHE_ESTIMATES.clear()
            ds._INDEX_CACHE_BYTES = 0
