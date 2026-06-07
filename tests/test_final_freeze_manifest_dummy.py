from pathlib import Path

from deep_oc_sort_3d.final_freeze.baseline_manifest import file_manifest_row, manifest_rows_for_baseline
from deep_oc_sort_3d.final_freeze.freeze_io import NOT_AVAILABLE


def test_final_freeze_file_manifest_counts_track1_rows(tmp_path):
    track1 = tmp_path / "track1.txt"
    track1.write_text("23 0 1 0 0 0 0 1 1 1 0\n23 0 1 1 0 0 0 1 1 1 0\n", encoding="utf-8")

    row = file_manifest_row("v1", "submission", "track1", track1)

    assert row["exists"] is True
    assert row["track1_rows"] == 2
    assert row["sha256"] != NOT_AVAILABLE


def test_final_freeze_manifest_rows_for_baseline_handles_missing(tmp_path):
    missing = tmp_path / "missing.txt"
    baseline = {
        "name": "dummy",
        "role": "diagnostic",
        "files": [{"kind": "track1", "path": missing}],
    }

    rows = manifest_rows_for_baseline(baseline)

    assert len(rows) == 1
    assert rows[0]["exists"] is False
    assert rows[0]["sha256"] == NOT_AVAILABLE

