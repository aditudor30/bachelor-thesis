import csv
import json

from deep_oc_sort_3d.person_association.person_association_metrics import collect_person_association_metrics


def test_person_association_metrics_reads_dummy_exports(tmp_path):
    final_root = tmp_path / "final_export"
    generic_dir = final_root / "generic_tracking_export" / "test"
    frame_dir = final_root / "frame_global_records" / "official_val" / "Warehouse_020"
    track1_root = tmp_path / "track1_submission"
    generic_dir.mkdir(parents=True)
    frame_dir.mkdir(parents=True)
    track1_root.mkdir(parents=True)
    generic_fields = [
        "scene_name",
        "camera_id",
        "frame_id",
        "global_track_id",
        "class_id",
        "class_name",
        "confidence",
        "x1",
        "y1",
        "x2",
        "y2",
        "w",
        "h",
        "center_x",
        "center_y",
        "center_z",
        "width_3d",
        "length_3d",
        "height_3d",
        "yaw",
    ]
    with (generic_dir / "Warehouse_023.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=generic_fields)
        writer.writeheader()
        writer.writerow(
            {
                "scene_name": "Warehouse_023",
                "camera_id": "Camera_0000",
                "frame_id": "0",
                "global_track_id": "1",
                "class_id": "0",
                "class_name": "Person",
                "confidence": "0.9",
                "x1": "0",
                "y1": "0",
                "x2": "10",
                "y2": "10",
                "w": "10",
                "h": "10",
                "center_x": "0",
                "center_y": "0",
                "center_z": "0",
                "width_3d": "1",
                "length_3d": "1",
                "height_3d": "2",
                "yaw": "0",
            }
        )
    frame_fields = [
        "scene_id",
        "scene_name",
        "split",
        "subset",
        "camera_id",
        "frame_id",
        "global_track_id",
        "local_track_id",
        "candidate_id",
        "detection_id",
        "class_id",
        "class_name",
        "confidence",
        "x1",
        "y1",
        "x2",
        "y2",
        "w",
        "h",
        "center_x",
        "center_y",
        "center_z",
        "width_3d",
        "length_3d",
        "height_3d",
        "yaw",
        "matched_gt_object_id",
        "matched_gt",
        "source",
    ]
    with (frame_dir / "Camera_0000_global_records.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=frame_fields)
        writer.writeheader()
        row = {field: "" for field in frame_fields}
        row.update(
            {
                "scene_id": "20",
                "scene_name": "Warehouse_020",
                "split": "val",
                "subset": "official_val",
                "camera_id": "Camera_0000",
                "frame_id": "0",
                "global_track_id": "1",
                "local_track_id": "1",
                "detection_id": "1",
                "class_id": "0",
                "class_name": "Person",
                "confidence": "0.9",
                "x1": "0",
                "y1": "0",
                "x2": "10",
                "y2": "10",
                "w": "10",
                "h": "10",
                "center_x": "0",
                "center_y": "0",
                "center_z": "0",
                "width_3d": "1",
                "length_3d": "1",
                "height_3d": "2",
                "yaw": "0",
                "matched_gt_object_id": "99",
                "matched_gt": "True",
                "source": "dummy",
            }
        )
        writer.writerow(row)
    (track1_root / "track1_export_summary.json").write_text(json.dumps({"rows_written": 1}), encoding="utf-8")
    (track1_root / "track1_validation_report.json").write_text(json.dumps({"status": "ok", "num_errors": 0}), encoding="utf-8")
    metrics = collect_person_association_metrics("dummy", final_root, track1_root)
    assert metrics["person_rows"] == 1
    assert metrics["non_person_rows"] == 0
    assert metrics["track1_rows"] == 1
    assert metrics["person_fragmentation_approx"] == 0

