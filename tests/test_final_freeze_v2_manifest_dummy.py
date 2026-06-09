from pathlib import Path

from deep_oc_sort_3d.final_freeze_v2.final_checksum_utils import compute_sha256
from deep_oc_sort_3d.final_freeze_v2.final_manifest_builder import artifact_row


def test_final_freeze_v2_manifest_and_checksum(tmp_path):
    path = tmp_path / "track1.txt"
    path.write_text("23 0 1 0 0 0 0 1 1 1 0\n", encoding="utf-8")
    row = artifact_row(path, "track1", "v1", "baseline", "desc")
    assert row["exists"] is True
    assert row["file_size"] > 0
    assert len(compute_sha256(path)) == 64

