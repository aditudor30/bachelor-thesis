import csv

from deep_oc_sort_3d.final_export.track1_dedup_audit import audit_generic_to_track1_dedup


def test_track1_dedup_audit_counts_duplicate_camera_observations(tmp_path):
    generic_root = tmp_path / "generic" / "test"
    generic_root.mkdir(parents=True)
    _write_generic(generic_root / "Warehouse_023.csv")
    track1_path = tmp_path / "track1.txt"
    track1_path.write_text("23 0 10 0 1.0 2.0 3.0 1.0 2.0 3.0 0.1\n", encoding="utf-8")

    report = audit_generic_to_track1_dedup(generic_root, track1_path, show_progress=False)

    assert report["generic_rows_total"] == 2
    assert report["official_rows_total"] == 1
    assert report["duplicate_rows_removed_estimated"] == 1
    assert report["duplicate_keys_count"] == 1


def _write_generic(path):
    fields = [
        "scene_name",
        "camera_id",
        "frame_id",
        "global_track_id",
        "class_id",
        "confidence",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerow(_row("Camera_0000", "0.8"))
        writer.writerow(_row("Camera_0001", "0.9"))


def _row(camera_id, confidence):
    return {
        "scene_name": "Warehouse_023",
        "camera_id": camera_id,
        "frame_id": "0",
        "global_track_id": "10",
        "class_id": "0",
        "confidence": confidence,
    }
