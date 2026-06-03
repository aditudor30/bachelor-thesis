from deep_oc_sort_3d.final_export.track1_schema_discovery import discover_track1_schema


def test_track1_schema_discovery_finds_text_matches(tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "challenge.md").write_text(
        "Track 1 submission details are pending.\n",
        encoding="utf-8",
    )

    report = discover_track1_schema(tmp_path, show_progress=False)

    assert len(report["matches"]) >= 1
    matched_terms = set([item["matched_term"] for item in report["matches"]])
    assert matched_terms.intersection(set(["Track 1", "track 1", "submission"]))
    assert set([item["file_path"] for item in report["matches"]]) == set(["docs/challenge.md"])
    assert set([item["line_number"] for item in report["matches"]]) == set([1])
    assert not report["found"]


def test_track1_schema_discovery_ignores_large_and_binary_files(tmp_path):
    (tmp_path / "large.md").write_text("Track 1\n" * 400000, encoding="utf-8")
    (tmp_path / "binary.txt").write_bytes(b"\x00track1.txt")

    report = discover_track1_schema(tmp_path, max_file_size_mb=1, show_progress=False)

    assert report["matches"] == []
    assert not report["found"]
