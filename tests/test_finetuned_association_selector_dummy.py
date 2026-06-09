from deep_oc_sort_3d.reid_finetuned_association.finetuned_association_selector import select_finetuned_reid_variant


def test_selector_picks_safe_fragmentation_gain():
    comparison = {
        "baseline": {"run_name": "v2_current"},
        "runs": [
            {
                "run_name": "threshold_075",
                "track1_validation_errors": 0,
                "track1_validation_status": "ok",
                "vs_v2_non_person_rows_delta": 0,
                "vs_v2_global_purity_mean_delta": -0.001,
                "vs_v2_false_merge_rate_delta": 0.002,
                "vs_v2_person_fragmentation_approx_delta": -80,
                "vs_v2_track1_rows_delta": 10,
            },
            {
                "run_name": "threshold_065",
                "track1_validation_errors": 0,
                "track1_validation_status": "ok",
                "vs_v2_non_person_rows_delta": 0,
                "vs_v2_global_purity_mean_delta": -0.02,
                "vs_v2_false_merge_rate_delta": 0.03,
                "vs_v2_person_fragmentation_approx_delta": -200,
                "vs_v2_track1_rows_delta": 10,
            },
        ],
    }
    selected = select_finetuned_reid_variant(
        comparison,
        {
            "max_allowed_false_merge_rate_delta": 0.01,
            "max_allowed_purity_drop": 0.01,
            "min_person_fragmentation_reduction_for_clear_gain": 50,
            "require_track1_valid": True,
            "require_non_person_unchanged": True,
        },
    )
    assert selected["best_run"] == "threshold_075"
    assert selected["verdict"] == "finetuned_reid_association_improves_v2"


def test_selector_reports_no_clear_gain_for_no_safe_runs():
    comparison = {
        "baseline": {"run_name": "v2_current"},
        "runs": [
            {
                "run_name": "threshold_080",
                "track1_validation_errors": 0,
                "track1_validation_status": "ok",
                "vs_v2_non_person_rows_delta": 0,
                "vs_v2_global_purity_mean_delta": 0.0,
                "vs_v2_false_merge_rate_delta": 0.0,
                "vs_v2_person_fragmentation_approx_delta": 0,
            }
        ],
    }
    selected = select_finetuned_reid_variant(comparison, {"require_track1_valid": True, "require_non_person_unchanged": True})
    assert selected["best_run"] is None
    assert selected["verdict"] == "finetuned_reid_valid_but_no_clear_gain"
