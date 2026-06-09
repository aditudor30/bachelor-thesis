from deep_oc_sort_3d.reid_visual_decision.visual_decision_metrics import decide_final_variant, summarize_visual_rows


def test_reid_visual_decision_metrics_selects_combined_when_visuals_good():
    rows = [
        {"variant": "combined_safe_080", "auto_label": "likely_good", "risk_score": "0.1"},
        {"variant": "combined_safe_080", "auto_label": "ambiguous", "risk_score": "0.3"},
    ]
    summary = summarize_visual_rows(rows)
    decision = decide_final_variant(summary, {"best_run": "combined_safe_080"})
    assert decision["final_verdict"] == "combined_safe_080_keep_as_experimental_final"

