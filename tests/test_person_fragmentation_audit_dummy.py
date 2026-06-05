import csv

from deep_oc_sort_3d.person_cleanup.person_fragmentation_audit import audit_person_fragmentation


def test_person_fragmentation_audit_counts_short_tracks(tmp_path):
    final_root = tmp_path / "final"
    generic = final_root / "generic_tracking_export" / "test"
    generic.mkdir(parents=True)
    fields = ["scene_name", "camera_id", "frame_id", "global_track_id", "class_id", "class_name", "confidence", "x1", "y1", "x2", "y2", "center_x", "center_y", "center_z"]
    rows = [
        {"scene_name": "Warehouse_023", "camera_id": "Camera_0000", "frame_id": "0", "global_track_id": "1", "class_id": "0", "class_name": "Person", "confidence": "0.01", "x1": "0", "y1": "0", "x2": "10", "y2": "10", "center_x": "0", "center_y": "0", "center_z": "0"},
        {"scene_name": "Warehouse_023", "camera_id": "Camera_0000", "frame_id": "1", "global_track_id": "2", "class_id": "0", "class_name": "Person", "confidence": "0.8", "x1": "20", "y1": "20", "x2": "30", "y2": "30", "center_x": "5", "center_y": "0", "center_z": "0"},
    ]
    with (generic / "Warehouse_023.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    summary = audit_person_fragmentation(final_root, tmp_path / "audit", {"class_id": 0}, progress=False)

    assert summary["person_generic_rows"] == 2
    assert summary["person_global_tracks"] == 2
    assert summary["singleton_tracks"] == 2
    assert (tmp_path / "audit" / "person_fragmentation_summary.json").exists()

