from deep_oc_sort_3d.reid_visual_decision.manual_review_writer import write_manual_review_sheet


def test_reid_visual_manual_review_writes_expected_columns(tmp_path):
    rows = [
        {
            "variant": "threshold_080",
            "merge_event_id": "e1",
            "auto_label": "ambiguous",
            "panel_path": "panel.png",
        }
    ]
    path = write_manual_review_sheet(rows, tmp_path, "threshold_080")
    text = path.read_text(encoding="utf-8")
    assert "human_label" in text
    assert "reviewer_notes" in text
