from deep_oc_sort_3d.final_export.track1_packaging import compute_file_sha256, package_track1_submission


def test_track1_packaging_creates_manifest_and_checksums(tmp_path):
    track1 = tmp_path / "track1.txt"
    track1.write_text("23 0 10 0 1.0 2.0 3.0 1.0 2.0 3.0 0.1\n", encoding="utf-8")
    config = tmp_path / "config.yaml"
    config.write_text("name: dummy\n", encoding="utf-8")
    report = tmp_path / "report.json"
    report.write_text('{"status": "ok"}\n', encoding="utf-8")
    package_root = tmp_path / "package"

    summary = package_track1_submission(
        track1_path=track1,
        output_package_root=package_root,
        config_paths=[config],
        reports=[report],
        baseline_name="baseline_v1_geometry_only",
        overwrite=False,
        make_zip=False,
        show_progress=False,
    )

    assert (package_root / "track1.txt").exists()
    assert (package_root / "configs" / "config.yaml").exists()
    assert (package_root / "reports" / "report.json").exists()
    assert (package_root / "manifest.json").exists()
    assert (package_root / "checksums.txt").exists()
    assert summary["package_root"] == str(package_root)
    assert compute_file_sha256(package_root / "track1.txt")
