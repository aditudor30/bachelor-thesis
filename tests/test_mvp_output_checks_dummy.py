import csv
import json

from deep_oc_sort_3d.scripts.check_mvp_outputs import check_mvp_outputs


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


def test_check_mvp_outputs_dummy_ok(tmp_path):
    config = _make_dummy_outputs(tmp_path, validation_errors=0)

    report = check_mvp_outputs(config, show_progress=False)

    assert report["status"] == "ok"
    assert report["errors"] == []
    assert report["summary"]["generic_export_scan"]["rows"] == 2


def test_check_mvp_outputs_dummy_flags_validation_errors(tmp_path):
    config = _make_dummy_outputs(tmp_path, validation_errors=2)

    report = check_mvp_outputs(config, show_progress=False)

    assert report["status"] == "error"
    assert "validation num_errors is 2" in report["errors"]


def _make_dummy_outputs(tmp_path, validation_errors):
    for name in [
        "pipeline",
        "local_tracks",
        "tracklets",
        "candidates",
        "candidates_clean",
        "global",
    ]:
        (tmp_path / name).mkdir()
    final_root = tmp_path / "final"
    (final_root / "summaries").mkdir(parents=True)
    (final_root / "validation").mkdir(parents=True)
    (final_root / "eval").mkdir(parents=True)
    generic_dir = final_root / "generic_tracking_export" / "test"
    generic_dir.mkdir(parents=True)

    _write_json(
        final_root / "summaries" / "propagation_summary.json",
        {
            "files": 1,
            "input_records": 4,
            "output_records": 4,
            "assigned_records": 2,
            "unassigned_records": 2,
            "assignment_ratio": 0.5,
            "unique_global_tracks": 1,
        },
    )
    _write_json(
        final_root / "summaries" / "export_summary.json",
        {
            "files": 1,
            "rows_written": 2,
        },
    )
    _write_json(
        final_root / "validation" / "global_validation_summary.json",
        {
            "files": 2,
            "num_errors": validation_errors,
            "num_warnings": 0,
        },
    )
    _write_json(
        final_root / "eval" / "global_eval.json",
        {
            "num_records": 2,
            "global_id_purity_mean": 0.95,
            "fragmentation_approx": 0,
        },
    )
    _write_generic_csv(generic_dir / "Warehouse_023.csv")
    return {
        "paths": {
            "pipeline_run_root": str(tmp_path / "pipeline"),
            "local_tracks_root": str(tmp_path / "local_tracks"),
            "tracklets_root": str(tmp_path / "tracklets"),
            "mtmc_candidates_root": str(tmp_path / "candidates"),
            "motion_clean_candidates_root": str(tmp_path / "candidates_clean"),
            "global_mtmc_root": str(tmp_path / "global"),
            "final_export_root": str(final_root),
        },
        "subsets": {
            "test": {
                "split": "test",
                "scenes": ["Warehouse_023"],
            },
        },
        "classes": {0: "Person"},
        "final_export": {
            "official_track1_export": "todo_until_schema_confirmed",
        },
        "sanity_checks": {
            "min_assignment_ratio": 0.4,
            "min_global_purity": 0.9,
            "expected_generic_export_files": 1,
            "require_validation_errors_zero": True,
        },
    }


def _write_json(path, data):
    path.write_text(json.dumps(data), encoding="utf-8")


def _write_generic_csv(path):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=GENERIC_FIELDS)
        writer.writeheader()
        writer.writerow(_generic_row(0, 1))
        writer.writerow(_generic_row(1, 1))


def _generic_row(frame_id, global_track_id):
    return {
        "scene_name": "Warehouse_023",
        "camera_id": "Camera_0000",
        "frame_id": frame_id,
        "global_track_id": global_track_id,
        "class_id": 0,
        "class_name": "Person",
        "confidence": 0.9,
        "x1": 1.0,
        "y1": 2.0,
        "x2": 10.0,
        "y2": 20.0,
        "w": 9.0,
        "h": 18.0,
        "center_x": 0.0,
        "center_y": 0.0,
        "center_z": 0.0,
        "width_3d": 1.0,
        "length_3d": 1.0,
        "height_3d": 1.0,
        "yaw": 0.0,
    }
