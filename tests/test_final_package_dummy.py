from deep_oc_sort_3d.final_freeze.final_package import build_package


def test_final_package_copies_files_and_writes_manifest(tmp_path):
    source = tmp_path / "source.txt"
    source.write_text("hello\n", encoding="utf-8")
    package_root = tmp_path / "packages"
    spec = {
        "name": "dummy_package",
        "title": "Dummy Package",
        "description": "Dummy package description.",
        "files": [{"kind": "text", "source": source, "relative": "nested/source.txt"}],
    }

    result = build_package(spec, package_root, show_progress=False, overwrite=True)

    root = package_root / "dummy_package"
    assert result["name"] == "dummy_package"
    assert (root / "nested" / "source.txt").exists()
    assert (root / "manifest.json").exists()
    assert (root / "CHECKSUMS.sha256").exists()

