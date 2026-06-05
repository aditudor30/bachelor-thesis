import csv

from deep_oc_sort_3d.person_cleanup.person_export_policy import apply_person_cleanup_export_policy


GENERIC_FIELDS = [
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


def test_person_export_policy_preserves_non_person(tmp_path):
    source = tmp_path / "source"
    generic = source / "generic_tracking_export" / "test"
    generic.mkdir(parents=True)
    rows = [
        _row("1", "0", "Person", "0.01"),
        _row("2", "1", "Forklift", "0.01"),
        _row("3", "0", "Person", "0.80"),
    ]
    with (generic / "Warehouse_023.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=GENERIC_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    summary = apply_person_cleanup_export_policy(
        source,
        tmp_path / "out",
        {
            "pruning": {
                "enabled": True,
                "class_id": 0,
                "mode": "short_lowconf",
                "max_rows_per_track": 1,
                "mean_confidence_threshold": 0.03,
                "max_confidence_threshold": 0.08,
            },
            "selective_merge_safe": {"enabled": False},
        },
        show_progress=False,
    )

    with (tmp_path / "out" / "generic_tracking_export" / "test" / "Warehouse_023.csv").open("r", newline="", encoding="utf-8") as handle:
        kept = list(csv.DictReader(handle))
    assert summary["generic_report"]["non_person_rows_dropped"] == 0
    assert [row["global_track_id"] for row in kept] == ["2", "3"]


def _row(track_id, class_id, class_name, confidence):
    row = {field: "0" for field in GENERIC_FIELDS}
    row.update(
        {
            "scene_name": "Warehouse_023",
            "camera_id": "Camera_0000",
            "frame_id": "0",
            "global_track_id": track_id,
            "class_id": class_id,
            "class_name": class_name,
            "confidence": confidence,
            "x2": "10",
            "y2": "10",
            "w": "10",
            "h": "10",
            "width_3d": "1",
            "length_3d": "1",
            "height_3d": "1",
        }
    )
    return row

