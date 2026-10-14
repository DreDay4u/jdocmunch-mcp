"""Issue #95 QA-19 lifecycle-authority regressions."""

from __future__ import annotations

import json
import multiprocessing
from contextlib import contextmanager

import pytest

from jdocmunch_mcp.storage import retirements
from jdocmunch_mcp.storage import doc_store as storage_module
from jdocmunch_mcp.storage.doc_store import DocStore, RetirementConflict
from jdocmunch_mcp.tools import _worktree_corpus as wc
from tests import test_v1_110_0 as legacy


def _pair(tmp_path, monkeypatch):
    _, _, _, store_path = legacy._standard_pair(tmp_path, monkeypatch)
    return store_path, DocStore(base_path=str(store_path))


def _fingerprints(store):
    values = {
        "local/old": store.index_fingerprint("local", "old"),
        "local/modern": store.index_fingerprint("local", "modern"),
    }
    assert all(values.values())
    return values


def _publish(store_path, store, retained="local/modern"):
    return retirements.begin_retirement(
        str(store_path),
        "local",
        "old",
        retained=retained,
        fingerprints=_fingerprints(store),
        family="qa19",
    )


def _finish_retirement(store_path, publication):
    """Complete exactly ``publication``, refused unless it is still current."""
    return retirements.finish_retirement(
        str(store_path), "local", "old", publication_id=publication
    )


def _current_retirement_record(store_path):
    return retirements.retirement_record(str(store_path), "local", "old")


def _guarded_delete(store, fingerprints, publication, *, lock_wait=True):
    return store.delete_index(
        "local",
        "old",
        expected_fingerprints=fingerprints,
        retirement_publication=publication,
        lock_wait=lock_wait,
    )


def _publication_worker(store_path, retained, ready, start, output):
    store = DocStore(base_path=store_path)
    ready.set()
    if not start.wait(15):
        raise TimeoutError("publisher start barrier was not released")
    publication = retirements.begin_retirement(
        store_path,
        "local",
        "old",
        retained=retained,
        fingerprints=_fingerprints(store),
        family="qa19-process",
    )
    output.put(publication)


def test_proof_and_record_side_fingerprints_are_one_implementation(
    tmp_path, monkeypatch
):
    """Both sides of the proof must compute the identical token.

    ``DocStore.index_fingerprint`` captures the token and
    ``begin_retirement`` re-proves it under the record lock. If those ever
    disagreed by a byte, publication would refuse itself and every retirement
    would silently become ``record_unavailable``, a failure that no
    behavioural test reads as a fingerprint bug.
    """
    store_path, store = _pair(tmp_path, monkeypatch)

    for owner, name in (("local", "old"), ("local", "modern")):
        assert retirements._fingerprint_handle(
            str(store_path), f"{owner}/{name}"
        ) == store.index_fingerprint(owner, name)

    # Both report None for a handle with no stored monolith, rather than one
    # raising and the other returning a hash of something else.
    assert retirements._fingerprint_handle(
        str(store_path), "local/absent"
    ) is None
    assert store.index_fingerprint("local", "absent") is None


def test_traversing_handle_never_hashes_a_file_outside_the_store(
    tmp_path, monkeypatch
):
    """An escaping handle must not be hashed from wherever it points.

    Without component validation, ``../escape`` joins to a path OUTSIDE the
    store and the fingerprint is computed from that file, yielding a retirement
    proof token derived from something the store does not own. The planted file and
    the path assertion below keep this from passing vacuously: the naive join
    really does reach it.
    """
    store_path, _ = _pair(tmp_path, monkeypatch)
    outside = store_path.parent / "escape.json"
    outside.write_text("not part of this store", encoding="utf-8")
    assert (store_path / ".." / "escape.json").resolve() == outside.resolve()

    assert retirements._fingerprint_handle(
        str(store_path), "../escape"
    ) is None


@pytest.mark.parametrize(
    "handle",
    ["local/../../escape", "./local", "local/.", "noslash", ""],
)
def test_unsafe_handles_fail_closed(tmp_path, monkeypatch, handle):
    """The remaining malformed shapes never produce a proof token."""
    store_path, _ = _pair(tmp_path, monkeypatch)
    assert retirements._fingerprint_handle(str(store_path), handle) is None


def test_drive_relative_handle_never_escapes_the_store(tmp_path, monkeypatch):
    """Windows drive-relative syntax is traversal without a separator.

    ``C:..`` contains no separator and is neither ``.`` nor ``..``, so a
    hand-rolled "no separators, no dots" rule admits it, but Windows resolves
    it drive-relative, so the join walks out of the store and hashes whatever
    it lands on. Only the canonical component policy rejects it.
    """
    store_path, _ = _pair(tmp_path, monkeypatch)
    outside = store_path.parent / "escape.json"
    outside.write_text("not part of this store", encoding="utf-8")

    for handle in ("C:../escape", "D:/escape", "C:escape"):
        assert retirements._fingerprint_handle(
            str(store_path), handle
        ) is None, f"{handle!r} produced a proof token"


@pytest.mark.parametrize(
    "component,safe",
    [
        ("modern", True), ("my.repo", True), ("a-b_c.1", True),
        ("C:..", False), ("D:", False), ("..", False), ("."  , False),
        ("", False), ("a/b", False), ("a\\b", False), (None, False),
    ],
)
def test_component_policy_is_one_shared_rule(component, safe, tmp_path):
    """Store side and record side must agree on what is safe to join."""
    # tmp_path, not a relative name: DocStore creates its base directory, and a
    # relative one appeared as `unused/` in whatever directory pytest ran from.
    store = DocStore(base_path=str(tmp_path / "store"))
    assert retirements.is_safe_path_component(component) is safe
    if safe:
        assert store._safe_repo_component(component, "name") == component
    else:
        with pytest.raises(ValueError):
            store._safe_repo_component(component, "name")


def test_publication_receipt_is_unique_and_persisted(tmp_path, monkeypatch):
    store_path, store = _pair(tmp_path, monkeypatch)

    first = _publish(store_path, store)
    second = _publish(store_path, store)
    record = retirements.pending_retirement(str(store_path), "local", "old")

    assert isinstance(first, str) and first
    assert isinstance(second, str) and second
    assert first != second
    assert record["publication_id"] == second


def test_older_completion_cannot_remove_newer_publication(tmp_path, monkeypatch):
    store_path, store = _pair(tmp_path, monkeypatch)
    first = _publish(store_path, store)
    second = _publish(store_path, store)

    assert _finish_retirement(store_path, first) is False
    record = retirements.pending_retirement(str(store_path), "local", "old")
    assert record is not None
    publication_id = record.get("publication_id")
    assert isinstance(publication_id, str), (
        "retirement record lacks stable publication identity"
    )
    assert publication_id == second


def test_publication_fails_closed_when_record_lock_is_unavailable(
    tmp_path, monkeypatch
):
    store_path, store = _pair(tmp_path, monkeypatch)
    monkeypatch.setattr(retirements, "_acquire_fd", lambda *args, **kwargs: None)

    assert _publish(store_path, store) is None
    record_path = store_path / "local" / ".retirements" / "old.json"
    assert record_path.exists() is False


def test_publication_revalidates_fingerprints_after_lock_acquisition(
    tmp_path, monkeypatch
):
    store_path, store = _pair(tmp_path, monkeypatch)
    fingerprints = _fingerprints(store)
    retained_path = store._index_path("local", "modern")
    real_acquire = retirements._acquire_fd
    changed = False

    def mutate_before_lock(lock_path, blocking):
        nonlocal changed
        if not changed:
            changed = True
            data = json.loads(retained_path.read_text(encoding="utf-8"))
            data["source_dirty"] = not data.get("source_dirty", False)
            retained_path.write_text(
                json.dumps(data, separators=(",", ":")), encoding="utf-8"
            )
        return real_acquire(lock_path, blocking)

    monkeypatch.setattr(retirements, "_acquire_fd", mutate_before_lock)
    publication = retirements.begin_retirement(
        str(store_path),
        "local",
        "old",
        retained="local/modern",
        fingerprints=fingerprints,
        family="qa19",
    )

    assert changed is True
    assert publication is None
    assert retirements.retirement_record(
        str(store_path), "local", "old"
    ) is None


@pytest.mark.parametrize("omitted", ["local/old", "local/modern"])
def test_publication_requires_both_retirement_pair_fingerprints(
    tmp_path, monkeypatch, omitted
):
    store_path, store = _pair(tmp_path, monkeypatch)
    fingerprints = _fingerprints(store)
    fingerprints.pop(omitted)

    publication = retirements.begin_retirement(
        str(store_path),
        "local",
        "old",
        retained="local/modern",
        fingerprints=fingerprints,
        family="qa19",
    )

    assert publication is None
    assert retirements.retirement_record(
        str(store_path), "local", "old"
    ) is None


@pytest.mark.parametrize(
    "retained", ["local/absent", "local/old", "../escape"]
)
def test_publication_requires_a_distinct_retained_handle_bound_to_the_proof(
    tmp_path, monkeypatch, retained
):
    store_path, store = _pair(tmp_path, monkeypatch)

    publication = retirements.begin_retirement(
        str(store_path),
        "local",
        "old",
        retained=retained,
        fingerprints=_fingerprints(store),
        family="qa19",
    )

    assert publication is None
    assert retirements.retirement_record(
        str(store_path), "local", "old"
    ) is None


def test_publication_rejects_fingerprints_outside_the_retirement_pair(
    tmp_path, monkeypatch
):
    store_path, store = _pair(tmp_path, monkeypatch)
    other_path = store._index_path("local", "other")
    other_path.write_bytes(store._index_path("local", "modern").read_bytes())
    fingerprints = _fingerprints(store)
    fingerprints["local/other"] = store.index_fingerprint("local", "other")
    assert fingerprints["local/other"]

    publication = retirements.begin_retirement(
        str(store_path),
        "local",
        "old",
        retained="local/modern",
        fingerprints=fingerprints,
        family="qa19",
    )

    assert publication is None
    assert retirements.retirement_record(
        str(store_path), "local", "old"
    ) is None


@pytest.mark.parametrize(
    ("owner", "name", "escaped_record"),
    [
        ("../escape", "old", "outside-owner"),
        ("local", "../escape", "outside-name"),
    ],
)
def test_publication_rejects_an_unsafe_record_slot_before_path_creation(
    tmp_path, monkeypatch, owner, name, escaped_record
):
    store_path, store = _pair(tmp_path, monkeypatch)
    if escaped_record == "outside-owner":
        escaped_path = store_path.parent / "escape" / ".retirements" / "old.json"
    else:
        escaped_path = store_path / "local" / "escape.json"

    publication = retirements.begin_retirement(
        str(store_path),
        owner,
        name,
        retained="local/modern",
        fingerprints=_fingerprints(store),
        family="qa19",
    )

    assert publication is None
    assert escaped_path.exists() is False


def test_final_gate_rejects_a_malformed_durable_retirement_pair(
    tmp_path, monkeypatch
):
    store_path, store = _pair(tmp_path, monkeypatch)
    fingerprints = _fingerprints(store)
    publication = _publish(store_path, store)
    record_path = store_path / "local" / ".retirements" / "old.json"
    real_commit = store._commit_guarded_retirement

    def corrupt_pair_before_final_gate(*args, **kwargs):
        record = json.loads(record_path.read_text(encoding="utf-8"))
        record["retained"] = "local/absent"
        record_path.write_text(
            json.dumps(record, separators=(",", ":")), encoding="utf-8"
        )
        return real_commit(*args, **kwargs)

    monkeypatch.setattr(
        store, "_commit_guarded_retirement", corrupt_pair_before_final_gate
    )

    with pytest.raises(RetirementConflict):
        _guarded_delete(store, fingerprints, publication)

    assert store.load_index("local", "old") is not None


def test_authoritative_record_lock_never_yields_without_a_lock(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(retirements, "_acquire_fd", lambda *args, **kwargs: None)

    with pytest.raises(RuntimeError):
        with retirements.hold_record_lock(str(tmp_path), "local", "old"):
            pytest.fail("critical section entered without an authoritative lock")


def test_completion_lock_failure_preserves_the_publication(
    tmp_path, monkeypatch
):
    store_path, store = _pair(tmp_path, monkeypatch)
    publication = _publish(store_path, store)
    monkeypatch.setattr(retirements, "_acquire_fd", lambda *args, **kwargs: None)

    assert _finish_retirement(store_path, publication) is False
    current = _current_retirement_record(store_path)
    assert current is not None
    publication_id = current.get("publication_id")
    assert isinstance(publication_id, str), (
        "retirement record lacks stable publication identity"
    )
    assert publication_id == publication


def test_final_gate_lock_failure_keeps_the_retiring_monolith(
    tmp_path, monkeypatch
):
    store_path, store = _pair(tmp_path, monkeypatch)
    publication = _publish(store_path, store)
    monkeypatch.setattr(retirements, "_acquire_fd", lambda *args, **kwargs: None)

    with pytest.raises(RetirementConflict):
        _guarded_delete(
            store,
            _fingerprints(store),
            publication,
        )

    assert store.load_index("local", "old") is not None


def test_final_gate_requires_the_exact_current_publication(tmp_path, monkeypatch):
    store_path, store = _pair(tmp_path, monkeypatch)
    first = _publish(store_path, store)
    second = _publish(store_path, store)

    with pytest.raises(RetirementConflict):
        _guarded_delete(
            store,
            _fingerprints(store),
            first,
        )

    assert store.load_index("local", "old") is not None
    record = retirements.pending_retirement(str(store_path), "local", "old")
    assert record["publication_id"] == second


def test_guarded_delete_cannot_adopt_a_replacement_publication(
    tmp_path, monkeypatch
):
    """A receiptless guarded delete cannot borrow the slot's current authority.

    Two publications land for the same retiring handle, so the slot holds the
    replacement. A delete naming no receipt has no authority of its own and
    must not adopt the one it happens to find: it conflicts, and the newer
    publication survives untouched. Distinct from
    ``test_guarded_delete_requires_a_current_publication``, where the slot is
    empty and there is nothing to adopt.
    """
    store_path, store = _pair(tmp_path, monkeypatch)
    _publish(store_path, store)
    current_publication = _publish(store_path, store)

    with pytest.raises(RetirementConflict):
        store.delete_index(
            "local",
            "old",
            expected_fingerprints=_fingerprints(store),
            lock_wait=True,
        )

    assert store.load_index("local", "old") is not None
    record = retirements.retirement_record(
        str(store_path), "local", "old"
    )
    assert record["publication_id"] == current_publication


def test_guarded_delete_requires_a_current_publication(tmp_path, monkeypatch):
    store_path, store = _pair(tmp_path, monkeypatch)
    fingerprints = _fingerprints(store)
    publication = _publish(store_path, store)
    record_path = store_path / "local" / ".retirements" / "old.json"
    record_path.unlink()
    outcome = {}

    with pytest.raises(RetirementConflict):
        store.delete_index(
            "local",
            "old",
            expected_fingerprints=fingerprints,
            retirement_publication=publication,
            outcome=outcome,
            lock_wait=True,
        )

    assert store.load_index("local", "old") is not None
    assert outcome == {}
    assert retirements.retirement_record(
        str(store_path), "local", "old"
    ) is None


def test_empty_expected_fingerprints_never_authorize_a_delete(
    tmp_path, monkeypatch
):
    """An empty proof asserts nothing, receipt or no receipt.

    Two distinct refusals, and the second is the one that matters. Without a
    receipt the call is rejected for the missing receipt. WITH a valid
    receipt, an empty map must still be refused; otherwise the receipt is a
    password rather than a proof, and its holder can authorize a commit while
    asserting nothing about either participant.
    """
    store_path, store = _pair(tmp_path, monkeypatch)

    # No receipt.
    with pytest.raises(RetirementConflict):
        store.delete_index(
            "local", "old", expected_fingerprints={}, lock_wait=False
        )
    assert store.load_index("local", "old") is not None

    # Valid receipt, empty proof. Red against 914a285, which returned True
    # and removed the retiring index.
    publication = _publish(store_path, store)
    assert publication
    with pytest.raises(RetirementConflict):
        store.delete_index(
            "local", "old",
            expected_fingerprints={},
            retirement_publication=publication,
            lock_wait=True,
        )
    assert store.load_index("local", "old") is not None
    assert store.load_index("local", "modern") is not None

    # Omitting the argument entirely still selects the unguarded path.
    assert store.delete_index("local", "old", lock_wait=False) is True


def test_partial_proof_cannot_hide_a_changed_retained_peer(
    tmp_path, monkeypatch
):
    """The gate verifies the DURABLE proof, not the caller's copy of it.

    The publication records fingerprints for both participants. A caller that
    supplies only the retiring handle's fingerprint would, if the gate trusted
    the argument, commit without ever looking at the retained peer, so a peer
    that changed after publication goes unnoticed and AC-02 is violated.

    The peer is mutated on disk rather than through ``save_index`` so no
    record-voiding path runs: the publication stays current and the ONLY thing
    standing between the caller and an unauthorized commit is proof binding.
    Red against 914a285, which returned True.
    """
    store_path, store = _pair(tmp_path, monkeypatch)
    fingerprints = _fingerprints(store)
    publication = retirements.begin_retirement(
        str(store_path), "local", "old",
        retained="local/modern", fingerprints=fingerprints, family="qa19",
    )
    assert publication

    retained_path = store._index_path("local", "modern")
    data = json.loads(retained_path.read_text(encoding="utf-8"))
    data["source_dirty"] = not data.get("source_dirty", False)
    retained_path.write_text(
        json.dumps(data, separators=(",", ":")), encoding="utf-8"
    )
    assert store.index_fingerprint("local", "modern") != (
        fingerprints["local/modern"]
    )
    assert retirements.retirement_record(
        str(store_path), "local", "old"
    )["publication_id"] == publication

    with pytest.raises(RetirementConflict):
        store.delete_index(
            "local", "old",
            expected_fingerprints={"local/old": fingerprints["local/old"]},
            retirement_publication=publication,
            lock_wait=True,
        )

    assert store.load_index("local", "old") is not None


def test_altered_proof_values_are_refused(tmp_path, monkeypatch):
    """A map with the right keys but a wrong value is not the published one."""
    store_path, store = _pair(tmp_path, monkeypatch)
    fingerprints = _fingerprints(store)
    publication = _publish(store_path, store)
    assert publication

    altered = dict(fingerprints)
    altered["local/modern"] = "0" * 64

    with pytest.raises(RetirementConflict):
        store.delete_index(
            "local", "old",
            expected_fingerprints=altered,
            retirement_publication=publication,
            lock_wait=True,
        )

    assert store.load_index("local", "old") is not None
    assert store.load_index("local", "modern") is not None


def test_guarded_delete_rejects_an_unreadable_publication(
    tmp_path, monkeypatch
):
    store_path, store = _pair(tmp_path, monkeypatch)
    fingerprints = _fingerprints(store)
    publication = _publish(store_path, store)
    record_path = store_path / "local" / ".retirements" / "old.json"
    record_path.write_text("{not-json", encoding="utf-8")
    outcome = {}

    with pytest.raises(RetirementConflict):
        store.delete_index(
            "local",
            "old",
            expected_fingerprints=fingerprints,
            retirement_publication=publication,
            outcome=outcome,
            lock_wait=True,
        )

    assert store.load_index("local", "old") is not None
    assert outcome == {}
    assert record_path.read_text(encoding="utf-8") == "{not-json"


def test_publication_fails_closed_without_a_supported_lock_primitive(
    tmp_path, monkeypatch
):
    store_path, store = _pair(tmp_path, monkeypatch)
    monkeypatch.setattr(retirements, "fcntl", None)
    monkeypatch.setattr(retirements, "msvcrt", None)

    assert _publish(store_path, store) is None
    assert retirements.retirement_record(
        str(store_path), "local", "old"
    ) is None


def test_completion_unlink_failure_reports_retired_with_pending_cleanup(
    tmp_path, monkeypatch
):
    _, worktree, _, store_path = legacy._standard_pair(
        tmp_path, monkeypatch
    )
    record_path = store_path / "local" / ".retirements" / "old.json"
    real_unlink = type(record_path).unlink

    def fail_record_unlink(path, *args, **kwargs):
        if path == record_path:
            raise OSError("injected publication completion failure")
        return real_unlink(path, *args, **kwargs)

    monkeypatch.setattr(type(record_path), "unlink", fail_record_unlink)
    result = legacy._index(
        worktree / "docs",
        "old",
        store_path,
        legacy_reconcile="apply",
    )

    block = result["legacy_reconciliation"]
    assert block["reason_code"] == wc.REASON_LEGACY_RECONCILED
    cleanup_schema = storage_module.RETIREMENT_CLEANUP_OUTCOME_SCHEMA
    assert set(cleanup_schema) == {
        "retirement_cleanup_pending",
        "retirement_cleanup_record_state",
        "retirement_cleanup_owned",
    }
    assert {field: block[field] for field in cleanup_schema} == {
        "retirement_cleanup_pending": True,
        "retirement_cleanup_record_state": "readable",
        "retirement_cleanup_owned": True,
    }
    record = retirements.pending_retirement(
        str(store_path), "local", "old"
    )
    assert isinstance(record["publication_id"], str)
    assert record["publication_id"]
    store = DocStore(base_path=str(store_path))
    assert store.load_index("local", "old") is None
    assert store.load_index("local", "modern") is not None


def test_conflict_cleanup_failure_uses_precommit_pending_signal(
    tmp_path, monkeypatch
):
    _, worktree, _, store_path = legacy._standard_pair(
        tmp_path, monkeypatch
    )
    record_path = (
        store_path / "local" / ".retirements" / "old.json"
    )
    real_unlink = type(record_path).unlink

    def refuse_guarded_delete(self, *args, **kwargs):
        raise RetirementConflict(["local/modern"])

    def fail_record_unlink(path, *args, **kwargs):
        if path == record_path:
            raise OSError("injected pre-commit record cleanup failure")
        return real_unlink(path, *args, **kwargs)

    monkeypatch.setattr(DocStore, "delete_index", refuse_guarded_delete)
    monkeypatch.setattr(type(record_path), "unlink", fail_record_unlink)

    result = legacy._index(
        worktree / "docs",
        "old",
        store_path,
        legacy_reconcile="apply",
    )

    block = result["legacy_reconciliation"]
    assert set(block) == {
        "state",
        "reason_code",
        "detail",
        "established_handle",
        "changed_handles",
        "pending_retirement",
    }
    assert block["reason_code"] == wc.REASON_LEGACY_CONFLICT
    assert block["pending_retirement"] is True
    cleanup_schema = storage_module.RETIREMENT_CLEANUP_OUTCOME_SCHEMA
    assert set(block).isdisjoint(cleanup_schema)
    record = retirements.pending_retirement(
        str(store_path), "local", "old"
    )
    assert record is not None
    assert isinstance(record.get("publication_id"), str)
    assert record_path.exists()
    store = DocStore(base_path=str(store_path))
    assert store.load_index("local", "old") is not None
    assert store.load_index("local", "modern") is not None


def test_unreadable_record_after_commit_is_disclosed_as_unowned(
    tmp_path, monkeypatch
):
    """Disclosure reports OBSERVED durable state, not what the commit assumed.

    The completion unlink fails and the record becomes unparseable in the same
    step, so the emitter cannot confirm the publication it just completed. It
    must say so, reporting ``unreadable`` and not owned, rather than the
    publication identity it was holding in memory.
    """
    _, worktree, _, store_path = legacy._standard_pair(
        tmp_path, monkeypatch
    )
    record_path = store_path / "local" / ".retirements" / "old.json"
    real_unlink = type(record_path).unlink

    def corrupt_then_fail_unlink(path, *args, **kwargs):
        if path == record_path:
            path.write_text("{truncated", encoding="utf-8")
            raise OSError("injected publication completion failure")
        return real_unlink(path, *args, **kwargs)

    monkeypatch.setattr(type(record_path), "unlink", corrupt_then_fail_unlink)

    result = legacy._index(
        worktree / "docs",
        "old",
        store_path,
        legacy_reconcile="apply",
    )

    block = result["legacy_reconciliation"]
    assert block["reason_code"] == wc.REASON_LEGACY_RECONCILED
    assert block["retirement_cleanup_pending"] is True
    assert block["retirement_cleanup_record_state"] == "unreadable"
    assert block["retirement_cleanup_owned"] is False
    # The commit itself still stands: primary gone, retained peer intact.
    store = DocStore(base_path=str(store_path))
    assert store.load_index("local", "old") is None
    assert store.load_index("local", "modern") is not None


def test_pending_read_reports_record_when_stale_cleanup_unlink_fails(
    tmp_path, monkeypatch
):
    store_path, store = _pair(tmp_path, monkeypatch)
    publication = _publish(store_path, store)
    record_path = store_path / "local" / ".retirements" / "old.json"
    store._index_path("local", "old").unlink()
    real_unlink = type(record_path).unlink

    def fail_record_unlink(path, *args, **kwargs):
        if path == record_path:
            raise OSError("injected stale-record cleanup failure")
        return real_unlink(path, *args, **kwargs)

    monkeypatch.setattr(type(record_path), "unlink", fail_record_unlink)

    record = retirements.pending_retirement(
        str(store_path), "local", "old"
    )
    assert record is not None
    assert record["publication_id"] == publication
    assert record_path.exists()


def test_fingerprints_are_reproved_after_the_retained_gate(
    tmp_path, monkeypatch
):
    store_path, store = _pair(tmp_path, monkeypatch)
    publication = _publish(store_path, store)
    fingerprints = _fingerprints(store)
    retained_path = store._index_path("local", "modern")

    @contextmanager
    def mutate_after_gate(retained, owner, name):
        data = json.loads(retained_path.read_text(encoding="utf-8"))
        data["source_dirty"] = not data.get("source_dirty", False)
        retained_path.write_text(
            json.dumps(data, separators=(",", ":")), encoding="utf-8"
        )
        yield True

    monkeypatch.setattr(store, "_gate_retained_handle", mutate_after_gate)

    with pytest.raises(RetirementConflict):
        _guarded_delete(
            store,
            fingerprints,
            publication,
        )

    assert store.load_index("local", "old") is not None


def test_reverse_scan_revalidates_candidate_under_its_lock(tmp_path, monkeypatch):
    store_path, store = _pair(tmp_path, monkeypatch)
    first = _publish(store_path, store)
    record_path = store_path / "local" / ".retirements" / "old.json"
    replacement = json.loads(record_path.read_text(encoding="utf-8"))
    replacement["publication_id"] = f"{first}-replacement"
    replacement["retained"] = "local/other"
    real_acquire = retirements._acquire_fd
    replaced = False

    def replace_before_lock(lock_path, blocking):
        nonlocal replaced
        if not replaced:
            replaced = True
            record_path.write_text(
                json.dumps(replacement, separators=(",", ":")),
                encoding="utf-8",
            )
        return real_acquire(lock_path, blocking)

    monkeypatch.setattr(retirements, "_acquire_fd", replace_before_lock)
    retirements.void_retirements_referencing(
        str(store_path), "local/modern"
    )

    assert replaced is True
    assert record_path.exists(), (
        "reverse scan removed a replacement retirement publication"
    )
    current = json.loads(record_path.read_text(encoding="utf-8"))
    assert current["publication_id"] == replacement["publication_id"]


def test_stale_self_healing_revalidates_missing_index_under_lock(
    tmp_path, monkeypatch
):
    store_path, store = _pair(tmp_path, monkeypatch)
    first = _publish(store_path, store)
    record_path = store_path / "local" / ".retirements" / "old.json"
    retiring_path = store._index_path("local", "old")
    retiring_bytes = retiring_path.read_bytes()
    retiring_path.unlink()
    replacement = json.loads(record_path.read_text(encoding="utf-8"))
    replacement["publication_id"] = f"{first}-replacement"
    real_acquire = retirements._acquire_fd
    replaced = False

    def recreate_before_lock(lock_path, blocking):
        nonlocal replaced
        if not replaced:
            replaced = True
            retiring_path.write_bytes(retiring_bytes)
            record_path.write_text(
                json.dumps(replacement, separators=(",", ":")),
                encoding="utf-8",
            )
        return real_acquire(lock_path, blocking)

    monkeypatch.setattr(retirements, "_acquire_fd", recreate_before_lock)
    current = retirements.pending_retirement(str(store_path), "local", "old")

    assert replaced is True
    assert current["publication_id"] == replacement["publication_id"]
    assert record_path.exists()


def test_cross_process_older_cleanup_cannot_remove_winning_publication(
    tmp_path, monkeypatch
):
    store_path, _ = _pair(tmp_path, monkeypatch)
    context = multiprocessing.get_context("spawn")
    start = context.Event()
    ready = [context.Event(), context.Event()]
    output = context.Queue()
    publishers = [
        context.Process(
            target=_publication_worker,
            args=(str(store_path), retained, signal, start, output),
        )
        for retained, signal in zip(
            ("local/modern", "local/modern"), ready, strict=True
        )
    ]

    for process in publishers:
        process.start()
    assert all(signal.wait(15) for signal in ready)
    start.set()
    publications = [output.get(timeout=15), output.get(timeout=15)]
    for process in publishers:
        process.join(15)
        assert process.exitcode == 0

    record = retirements.pending_retirement(str(store_path), "local", "old")
    assert record is not None
    winner = record.get("publication_id")
    if not isinstance(winner, str):
        assert _finish_retirement(store_path, publications[0]) is False
        pytest.fail(
            "retirement completion lacked exact publication ownership"
        )
    loser = next(publication for publication in publications if publication != winner)
    assert _finish_retirement(store_path, loser) is False
    current = retirements.pending_retirement(
        str(store_path), "local", "old"
    )
    assert current is not None
    publication_id = current.get("publication_id")
    assert isinstance(publication_id, str), (
        "retirement record lacks stable publication identity"
    )
    assert publication_id == winner
