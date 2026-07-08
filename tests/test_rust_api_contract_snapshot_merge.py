"""Unit tests for the snapshot writer's merge behavior (sc-8862).

These are deliberately *not* marked ``parity`` — they exercise the pure
``write_updated_snapshots`` merge logic without spinning up the Rust API, so
they run in the default CI lane (``-m "not e2e and not parity"``) and guard
against the regression where a filtered ``UPDATE_SNAPSHOTS=1`` run clobbered
every untouched snapshot.
"""

from __future__ import annotations

import json

import test_rust_api_contract_snapshots as mod


def _write_snapshots(path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def test_filtered_update_preserves_untouched_snapshots(monkeypatch, tmp_path):
    snapshot_path = tmp_path / "snapshots.json"
    # Pre-existing on-disk snapshots, as if written by a prior full run.
    _write_snapshots(
        snapshot_path,
        {
            "character contract": {"kind": "character"},
            "timeline contract": {"kind": "timeline"},
            "worker contract": {"kind": "worker"},
        },
    )

    # Simulate `UPDATE_SNAPSHOTS=1 pytest -k character`: only the character
    # label is accumulated during the run.
    monkeypatch.setattr(mod, "SNAPSHOT_PATH", snapshot_path)
    monkeypatch.setattr(mod, "UPDATE_SNAPSHOTS", True)
    monkeypatch.setattr(
        mod,
        "UPDATED_SNAPSHOTS",
        {"character contract": {"kind": "character", "rev": 2}},
    )

    mod.write_updated_snapshots()

    result = json.loads(snapshot_path.read_text(encoding="utf-8"))
    # Touched label is updated in place.
    assert result["character contract"] == {"kind": "character", "rev": 2}
    # Untouched labels are preserved rather than clobbered.
    assert result["timeline contract"] == {"kind": "timeline"}
    assert result["worker contract"] == {"kind": "worker"}


def test_first_ever_generation_starts_from_empty(monkeypatch, tmp_path):
    # No file on disk yet: the writer must create it from the run's labels.
    snapshot_path = tmp_path / "nested" / "snapshots.json"
    assert not snapshot_path.exists()

    monkeypatch.setattr(mod, "SNAPSHOT_PATH", snapshot_path)
    monkeypatch.setattr(mod, "UPDATE_SNAPSHOTS", True)
    monkeypatch.setattr(mod, "UPDATED_SNAPSHOTS", {"only label": {"kind": "only"}})

    mod.write_updated_snapshots()

    result = json.loads(snapshot_path.read_text(encoding="utf-8"))
    assert result == {"only label": {"kind": "only"}}


def test_writer_is_noop_when_update_disabled(monkeypatch, tmp_path):
    snapshot_path = tmp_path / "snapshots.json"
    _write_snapshots(snapshot_path, {"existing": {"kind": "existing"}})

    monkeypatch.setattr(mod, "SNAPSHOT_PATH", snapshot_path)
    monkeypatch.setattr(mod, "UPDATE_SNAPSHOTS", False)
    monkeypatch.setattr(mod, "UPDATED_SNAPSHOTS", {"new": {"kind": "new"}})

    mod.write_updated_snapshots()

    # Unchanged: the writer returns early when UPDATE_SNAPSHOTS is off.
    result = json.loads(snapshot_path.read_text(encoding="utf-8"))
    assert result == {"existing": {"kind": "existing"}}


def test_output_is_sorted_and_stable(monkeypatch, tmp_path):
    snapshot_path = tmp_path / "snapshots.json"
    _write_snapshots(snapshot_path, {"zeta": {"kind": "z"}})

    monkeypatch.setattr(mod, "SNAPSHOT_PATH", snapshot_path)
    monkeypatch.setattr(mod, "UPDATE_SNAPSHOTS", True)
    monkeypatch.setattr(mod, "UPDATED_SNAPSHOTS", {"alpha": {"kind": "a"}})

    mod.write_updated_snapshots()

    text = snapshot_path.read_text(encoding="utf-8")
    # Keys are sorted for clean diffs: alpha before zeta, trailing newline.
    assert text.index('"alpha"') < text.index('"zeta"')
    assert text.endswith("}\n")
