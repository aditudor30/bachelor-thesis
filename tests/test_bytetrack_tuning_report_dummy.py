from deep_oc_sort_3d.bytetrack_tuning.tuning_report import write_markdown_report


def test_tuning_report_writes_selected_variant(tmp_path):
    result = {
        "selection": {
            "selected_variant": "dense_safe",
            "selected_metrics": {
                "track1_validation_errors": 0,
                "local_records_retention": 0.9,
                "gt_matched_retention": 0.95,
                "track1_rows_retention": 0.8,
                "multi_camera_tracks_retention": 0.7,
            },
            "variant_selection_rows": [
                {
                    "variant": "dense_safe",
                    "hard_criteria_met": True,
                    "local_records_retention": 0.9,
                    "gt_matched_retention": 0.95,
                    "track1_rows_retention": 0.8,
                    "multi_camera_tracks_retention": 0.7,
                    "selection_score": 12.0,
                }
            ],
            "verdict": {
                "label": "bytetrack_tuned_ready_for_full_submission_candidate",
                "reasons": ["hard_coverage_gates_passed"],
            },
        }
    }
    path = tmp_path / "report.md"
    write_markdown_report(result, path)
    text = path.read_text(encoding="utf-8")
    assert "dense_safe" in text
    assert "bytetrack_tuned_ready_for_full_submission_candidate" in text

