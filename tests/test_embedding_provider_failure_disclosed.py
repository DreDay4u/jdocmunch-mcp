"""`semantic_search: true` no longer stands alone when nothing was embedded.

Measured on a box whose FastEmbed model cache in the temp dir had been
emptied: the provider factory raised, a bare ``except`` returned None, every
section was indexed lexically, and `index_local` reported
`"semantic_search": true` with zero vectors written. The key keeps its 1.x
meaning (a provider is configured); `embedding_provider_error` and a warning
now carry the failure beside it.
"""

from __future__ import annotations

import pytest

from jdocmunch_mcp.embeddings import provider as P


class _Boom:
    def __init__(self):
        raise FileNotFoundError("model.onnx is missing")


class _Works:
    def embed_texts(self, texts, task_type="retrieval_document"):
        return [[0.1, 0.2, 0.3] for _ in texts]


class _RaisesOnEmbed:
    def embed_texts(self, texts, task_type="retrieval_document"):
        raise RuntimeError("rate limited")


@pytest.fixture
def fake_provider(monkeypatch):
    monkeypatch.setenv("JDOCMUNCH_EMBEDDING_PROVIDER", "fastembed")
    monkeypatch.setenv("JDOCMUNCH_SHARE_SAVINGS", "0")
    monkeypatch.setattr(P, "_PROVIDER_CACHE", {})
    monkeypatch.setattr(P, "_PROVIDER_ERRORS", {})

    def use(factory):
        monkeypatch.setitem(P._PROVIDER_FACTORIES, "fastembed", factory)

    return use


def _index(tmp_path):
    from jdocmunch_mcp.tools.index_local import index_local
    src = tmp_path / "docs"
    src.mkdir(exist_ok=True)
    (src / "a.md").write_text("# Widgets\n\nHow to configure widgets.\n", encoding="utf-8")
    return index_local(path=str(src), name="emb", use_ai_summaries=False,
                       storage_path=str(tmp_path / "store"), incremental=False)


def test_a_provider_that_fails_to_load_is_disclosed(tmp_path, fake_provider):
    fake_provider(_Boom)
    r = _index(tmp_path)
    assert r["success"]
    assert r["semantic_search"] is True  # 1.x meaning kept
    assert "model.onnx is missing" in r["embedding_provider_error"]
    assert any("produced no vectors" in w for w in r.get("warnings", []))


def test_an_embed_pass_that_raises_is_disclosed(tmp_path, fake_provider):
    fake_provider(_RaisesOnEmbed)
    r = _index(tmp_path)
    assert "rate limited" in r["embedding_provider_error"]


def test_a_working_provider_adds_nothing(tmp_path, fake_provider):
    fake_provider(_Works)
    r = _index(tmp_path)
    assert r["semantic_search"] is True
    assert "embedding_provider_error" not in r
    assert not any("produced no vectors" in w for w in r.get("warnings", []))


def test_a_later_success_clears_the_error(tmp_path, fake_provider):
    fake_provider(_Boom)
    P._get_provider()
    assert P.provider_error()
    fake_provider(_Works)
    assert P._get_provider() is not None
    assert P.provider_error() is None


def test_nothing_is_said_when_embeddings_are_off(tmp_path, fake_provider, monkeypatch):
    fake_provider(_Boom)
    monkeypatch.setenv("JDOCMUNCH_EMBEDDING_PROVIDER", "none")
    r = _index(tmp_path)
    assert r["semantic_search"] is False
    assert "embedding_provider_error" not in r


def test_a_long_reason_is_cut(fake_provider):
    class _Long:
        def __init__(self):
            raise ValueError("x" * 2000)
    fake_provider(_Long)
    P._get_provider()
    assert len(P.provider_error()) == P._ERROR_MAX_CHARS
