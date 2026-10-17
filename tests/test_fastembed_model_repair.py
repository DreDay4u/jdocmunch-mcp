"""`init` reports a broken FastEmbed model and repairs it only on consent.

Maintainer's ruling, 2026-10-17: "report the missing model with an opt-in
repair via init --with-embeddings, no silent download." Measured on the
development box: FastEmbed's model cache in the temp directory had lost
`model.onnx`, every load failed, and `init` still printed "Semantic search: on".

The repair downloads into jdocmunch's own folder, outside the temp directory.
The provider loads from there only when the model is present, so an install
whose temp-dir copy is fine is never re-downloaded.
"""

from __future__ import annotations

import sys
import types

import pytest

from jdocmunch_mcp.cli import embeddings_offer as eo
from jdocmunch_mcp.embeddings import provider as P


def _run(monkeypatch, *, problem_seq, interactive=False, with_embeddings=False,
         dry_run=False, answer=False):
    """Drive the offer with fastembed resolved and a scripted health check."""
    seq = list(problem_seq)
    downloads = []
    prompts = []
    lines = []
    monkeypatch.setattr(eo, "_resolved_provider", lambda: "fastembed")
    monkeypatch.setattr(eo, "_model_problem", lambda: seq.pop(0) if len(seq) > 1 else seq[0])
    monkeypatch.setattr(eo, "_download_model", lambda: downloads.append(1) or (True, ""))
    result = eo.offer_embeddings(
        interactive=interactive, with_embeddings=with_embeddings, dry_run=dry_run,
        prompt_yn=lambda text, default: prompts.append(text) or answer,
        out=lines.append,
    )
    return result, downloads, prompts, "\n".join(lines)


def test_a_healthy_model_reads_on(monkeypatch):
    ok, downloads, prompts, out = _run(monkeypatch, problem_seq=[None])
    assert ok is True and not downloads and not prompts
    assert "Semantic search: on (fastembed)" in out


def test_a_broken_model_is_reported_and_not_downloaded_unasked(monkeypatch):
    ok, downloads, prompts, out = _run(monkeypatch, problem_seq=["model.onnx missing"])
    assert ok is False and not downloads
    assert "won't load: model.onnx missing" in out
    assert "jdocmunch-mcp init --with-embeddings" in out
    assert "Semantic search: on" not in out


def test_dry_run_says_what_it_would_do(monkeypatch):
    ok, downloads, _p, out = _run(monkeypatch, problem_seq=["gone"], dry_run=True,
                                  with_embeddings=True)
    assert ok is False and not downloads
    assert "would download the model" in out


def test_with_embeddings_repairs(monkeypatch):
    ok, downloads, prompts, out = _run(monkeypatch, problem_seq=["gone", None],
                                       with_embeddings=True)
    assert ok is True and downloads == [1] and not prompts
    assert "Model repaired" in out


def test_a_yes_at_the_prompt_repairs_and_a_no_does_not(monkeypatch):
    ok, downloads, prompts, _ = _run(monkeypatch, problem_seq=["gone", None],
                                     interactive=True, answer=True)
    assert ok is True and downloads == [1] and "Repair it now?" in prompts[0]
    ok, downloads, _, out = _run(monkeypatch, problem_seq=["gone"],
                                 interactive=True, answer=False)
    assert ok is False and not downloads


def test_a_download_that_still_will_not_load_is_not_called_on(monkeypatch):
    ok, downloads, _, out = _run(monkeypatch, problem_seq=["gone", "still gone"],
                                 with_embeddings=True)
    assert ok is False and downloads == [1]
    assert "still won't load" in out


# --- where the provider loads from ----------------------------------------

@pytest.fixture
def clean_env(monkeypatch, tmp_path):
    monkeypatch.delenv("FASTEMBED_CACHE_PATH", raising=False)
    monkeypatch.delenv("JDOCMUNCH_FASTEMBED_CACHE", raising=False)
    monkeypatch.setattr(P, "_stable_fastembed_cache", lambda: tmp_path / "stable")
    return tmp_path


def _populate(root):
    snap = root / "models--qdrant--all-MiniLM-L6-v2-onnx" / "snapshots" / "abc"
    snap.mkdir(parents=True)
    (snap / "model.onnx").write_bytes(b"x")


def test_an_empty_stable_folder_leaves_fastembeds_default(clean_env, monkeypatch):
    monkeypatch.setattr(P, "_canonical_hub_model_id", lambda m: "qdrant/all-MiniLM-L6-v2-onnx")
    assert P._fastembed_cache_dir() is None


def test_a_populated_stable_folder_is_used(clean_env, monkeypatch):
    monkeypatch.setattr(P, "_canonical_hub_model_id", lambda m: "qdrant/all-MiniLM-L6-v2-onnx")
    _populate(clean_env / "stable")
    assert P._fastembed_cache_dir() == str(clean_env / "stable")


def test_fastembeds_own_variable_wins(clean_env, monkeypatch):
    monkeypatch.setattr(P, "_canonical_hub_model_id", lambda m: "qdrant/all-MiniLM-L6-v2-onnx")
    _populate(clean_env / "stable")
    monkeypatch.setenv("FASTEMBED_CACHE_PATH", str(clean_env / "theirs"))
    assert P._fastembed_cache_dir() is None


def test_an_explicit_jdocmunch_folder_is_used_even_empty(monkeypatch, tmp_path):
    monkeypatch.delenv("FASTEMBED_CACHE_PATH", raising=False)
    monkeypatch.setenv("JDOCMUNCH_FASTEMBED_CACHE", str(tmp_path / "mine"))
    assert P._fastembed_cache_dir() == str(tmp_path / "mine")


def test_the_provider_passes_folder_and_local_only(monkeypatch):
    seen = {}

    class _TE:
        def __init__(self, model_name, **kw):
            seen.update(kw, model_name=model_name)

    monkeypatch.setitem(sys.modules, "fastembed", types.SimpleNamespace(TextEmbedding=_TE))
    monkeypatch.setattr(P, "_fastembed_cache_dir", lambda: None)
    P._FastEmbedProvider()
    assert "cache_dir" not in seen and "local_files_only" not in seen  # the old call, exactly
    P._FastEmbedProvider(cache_dir="/x", local_files_only=True)
    assert seen["cache_dir"] == "/x" and seen["local_files_only"] is True


def test_the_health_check_never_downloads(monkeypatch):
    calls = {}

    class _TE:
        def __init__(self, model_name, **kw):
            calls.update(kw)
            raise ValueError("Could not find model in cache_dir")

    monkeypatch.setitem(sys.modules, "fastembed", types.SimpleNamespace(TextEmbedding=_TE))
    problem = P.fastembed_model_problem()
    assert calls.get("local_files_only") is True
    assert "Could not find model" in problem


def test_the_index_warning_names_the_repair_for_fastembed(monkeypatch):
    from jdocmunch_mcp.tools._embedding_coverage import attach_provider_failure
    monkeypatch.setattr(P, "provider_error", lambda: "NoSuchFile: model.onnx")
    monkeypatch.setattr(P, "get_provider_name", lambda: "fastembed")
    warnings: list = []
    result = attach_provider_failure({"semantic_search": True}, warnings)
    assert result["embedding_provider_error"] == "NoSuchFile: model.onnx"
    assert "init --with-embeddings" in warnings[0]
