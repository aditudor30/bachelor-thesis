from deep_oc_sort_3d.final_export.track1_final_checks import read_track1_txt, validate_track1_rows


def test_track1_final_checks_valid_rows(tmp_path):
    path = tmp_path / "track1.txt"
    path.write_text(
        "23 0 10 0 1.0 2.0 3.0 1.0 2.0 3.0 0.1\n"
        "23 0 10 1 1.1 2.1 3.1 1.0 2.0 3.0 0.1\n",
        encoding="utf-8",
    )

    report = validate_track1_rows(
        read_track1_txt(path),
        expected_scene_ids=[23],
        valid_class_ids=[0],
        show_progress=False,
    )

    assert report["status"] == "ok"
    assert report["num_errors"] == 0
    assert report["distribution"]["per_scene_rows"]["23"] == 2


def test_track1_final_checks_duplicate_key(tmp_path):
    path = tmp_path / "track1.txt"
    path.write_text(
        "23 0 10 0 1.0 2.0 3.0 1.0 2.0 3.0 0.1\n"
        "23 0 10 0 1.2 2.2 3.2 1.0 2.0 3.0 0.1\n",
        encoding="utf-8",
    )

    report = validate_track1_rows(read_track1_txt(path), expected_scene_ids=[23], valid_class_ids=[0], show_progress=False)

    assert report["status"] == "error"
    assert report["checks"]["duplicate_key_count"] == 1


def test_track1_final_checks_negative_dimension_and_nan(tmp_path):
    path = tmp_path / "track1.txt"
    path.write_text(
        "23 0 10 0 1.0 2.0 3.0 -1.0 2.0 3.0 0.1\n"
        "23 0 11 0 nan 2.0 3.0 1.0 2.0 3.0 0.1\n",
        encoding="utf-8",
    )

    report = validate_track1_rows(read_track1_txt(path), expected_scene_ids=[23], valid_class_ids=[0], show_progress=False)

    assert report["status"] == "error"
    assert report["checks"]["non_positive_dimensions"] == 1
    assert report["checks"]["nan_or_inf_values"] == 1
