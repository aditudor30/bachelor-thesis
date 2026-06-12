from deep_oc_sort_3d.bytetrack_local_pipeline.bytetrack_validation import validate_bytetrack_track1


def test_bytetrack_track1_validation_accepts_valid_dummy(tmp_path):
    track1 = tmp_path / "track1.txt"
    output_root = tmp_path / "validation"
    track1.write_text("23 0 1 0 1 2 3 1 1 1 0\n", encoding="utf-8")

    report = validate_bytetrack_track1(track1, output_root=output_root, progress=False)

    assert report["status"] == "ok"
    assert report["num_errors"] == 0
    assert (output_root / "track1_validation_summary.json").exists()
