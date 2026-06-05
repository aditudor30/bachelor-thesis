import csv

from deep_oc_sort_3d.global_tuning.compact_export_policy import compact_generic_export_file


def test_compact_export_policy_drops_short_low_conf_person_and_preserves_rare(tmp_path):
    input_path = tmp_path / "input.csv"
    output_path = tmp_path / "output.csv"
    fields = ["scene_name", "camera_id", "frame_id", "global_track_id", "class_id", "class_name", "confidence"]
    rows = [
        {"scene_name": "Warehouse_023", "camera_id": "Camera_0000", "frame_id": "0", "global_track_id": "1", "class_id": "0", "class_name": "Person", "confidence": "0.10"},
        {"scene_name": "Warehouse_023", "camera_id": "Camera_0000", "frame_id": "0", "global_track_id": "2", "class_id": "1", "class_name": "Forklift", "confidence": "0.05"},
        {"scene_name": "Warehouse_023", "camera_id": "Camera_0000", "frame_id": "0", "global_track_id": "3", "class_id": "0", "class_name": "Person", "confidence": "0.80"},
        {"scene_name": "Warehouse_023", "camera_id": "Camera_0000", "frame_id": "1", "global_track_id": "3", "class_id": "0", "class_name": "Person", "confidence": "0.80"},
        {"scene_name": "Warehouse_023", "camera_id": "Camera_0000", "frame_id": "2", "global_track_id": "3", "class_id": "0", "class_name": "Person", "confidence": "0.80"},
    ]
    with input_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    report = compact_generic_export_file(
        input_path,
        output_path,
        {
            "min_rows_per_track": 3,
            "min_mean_confidence": 0.2,
            "drop_single_frame_global_tracks": True,
            "single_frame_min_confidence": 0.35,
            "drop_low_conf_short_tracks": True,
            "protected_class_ids": [1],
        },
    )

    with output_path.open("r", newline="", encoding="utf-8") as handle:
        kept = list(csv.DictReader(handle))
    assert report["tracks_dropped"] == 1
    assert len(kept) == 4
    assert set([row["global_track_id"] for row in kept]) == set(["2", "3"])

