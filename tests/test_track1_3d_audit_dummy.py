from deep_oc_sort_3d.audit3d.track1_3d_audit import (
    compute_3d_field_stats,
    compute_per_class_3d_stats,
    compute_per_scene_3d_stats,
    detect_extreme_3d_values,
    read_track1_rows,
)


def test_track1_3d_audit_reads_and_summarizes_dummy_file(tmp_path):
    path = tmp_path / "track1.txt"
    path.write_text(
        "\n".join(
            [
                "23 0 10 0 1.0 2.0 3.0 0.8 1.2 1.7 0.0",
                "23 0 10 1 2.0 2.0 3.0 0.8 1.2 1.7 0.1",
                "24 1 11 0 4.0 5.0 6.0 1.0 2.0 1.5 0.2",
            ]
        ),
        encoding="utf-8",
    )

    rows = read_track1_rows(path, show_progress=False)
    summary = compute_3d_field_stats(rows)
    per_class = compute_per_class_3d_stats(rows)
    per_scene = compute_per_scene_3d_stats(rows)

    assert len(rows) == 3
    assert summary["field_stats"]["x"]["valid_count"] == 3
    assert summary["field_stats"]["x"]["median"] == 2.0
    assert len(per_class) == 2
    assert len(per_scene) == 2


def test_track1_3d_audit_flags_extreme_values(tmp_path):
    path = tmp_path / "track1.txt"
    path.write_text("23 0 10 0 999.0 2.0 3.0 -1.0 99.0 1.7 0.0\n", encoding="utf-8")

    rows = read_track1_rows(path, show_progress=False)
    extremes = detect_extreme_3d_values(rows, {"coordinate_abs_max": 100.0, "dimension_max": 20.0})

    issues = ";".join([str(row.get("issue")) for row in extremes])
    assert "x_abs_gt_100.0" in issues
    assert "width_non_positive" in issues
    assert "length_gt_20.0" in issues
