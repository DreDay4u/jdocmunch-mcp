"""The content mirror is written with one root resolve and one mkdir per parent.

`save_index` and `incremental_save` resolved the content root again for every
document and called `mkdir` once per document. On a 9,759-file corpus the
path checks fell from 5.4 s to 3.5 s. These tests pin what must not change:
every candidate path is still resolved (the escape guard), an unsafe path still
raises, and the bytes on disk are the same.
"""

from pathlib import Path

import pytest

from jdocmunch_mcp.storage.doc_store import DocStore


def test_mirror_writes_every_file_and_makes_each_parent_once(tmp_path, monkeypatch):
    store = DocStore(base_path=str(tmp_path / "store"))
    content_dir = tmp_path / "store" / "local" / "x"
    content_dir.mkdir(parents=True)
    raw = {f"docs/a{i}.md": f"# A{i}\n" for i in range(5)}
    raw.update({f"guide/b{i}.md": f"# B{i}\n" for i in range(3)})
    raw["top.md"] = "# Top\n"

    made = []
    real_mkdir = Path.mkdir

    def spy_mkdir(self, *a, **k):
        made.append(self.name)
        return real_mkdir(self, *a, **k)

    resolved = []
    real_safe = DocStore._safe_content_path

    def spy_safe(self, content_dir, rel, resolved_root=None):
        resolved.append((rel, resolved_root is not None))
        return real_safe(self, content_dir, rel, resolved_root)

    monkeypatch.setattr(Path, "mkdir", spy_mkdir)
    monkeypatch.setattr(DocStore, "_safe_content_path", spy_safe)
    store._write_content_mirror(content_dir, raw)

    for rel, text in raw.items():
        assert (content_dir / rel).read_bytes() == text.encode("utf-8")
    # One mkdir per distinct parent, not one per document.
    assert sorted(made) == ["docs", "guide", "x"]
    # Every candidate is still checked, each with the precomputed root.
    assert sorted(r for r, _ in resolved) == sorted(raw)
    assert all(has_root for _, has_root in resolved)


def test_mirror_still_refuses_a_path_that_escapes(tmp_path):
    store = DocStore(base_path=str(tmp_path / "store"))
    content_dir = tmp_path / "store" / "local" / "x"
    content_dir.mkdir(parents=True)
    with pytest.raises(ValueError, match="Unsafe doc path"):
        store._write_content_mirror(content_dir, {"../../escape.md": "# no\n"})
    assert not (tmp_path / "escape.md").exists()
