from deep_oc_sort_3d.final_freeze.metric_collector import build_track1_validation_summary, collect_pseudo3d_summary


def test_final_metric_collector_builds_track1_summary():
    rows = [
        {
            "variant_name": "V1",
            "role": "submission",
            "track1_valid": True,
            "track1_errors": 0,
            "track1_rows": 10,
        }
    ]

    summary = build_track1_validation_summary(rows)

    assert summary[0]["variant_name"] == "V1"
    assert summary[0]["track1_valid"] is True
    assert summary[0]["track1_rows"] == 10


def test_final_metric_collector_reads_pseudo3d_summary(tmp_path):
    root = tmp_path / "pseudo3d"
    root.mkdir()
    (root / "summary.json").write_text('{"pseudo3d_used_rate": 0.9, "fallback_original_used_rate": 0.1}', encoding="utf-8")
    config = {"paths": {"v2_final_export_root": str(root)}}

    summary = collect_pseudo3d_summary(config)

    assert summary["pseudo3d_used_rate"] == 0.9
    assert summary["fallback_original_used_rate"] == 0.1

