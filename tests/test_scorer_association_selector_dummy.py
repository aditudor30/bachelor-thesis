from deep_oc_sort_3d.learned_association_application.scorer_association_selector import select_variant


def test_selector_prefers_safe_fragmentation_gain():
    baseline = {
        "non_person_rows": 100,
        "person_false_merge_rate": 0.05,
        "person_purity_mean": 0.97,
        "person_fragmentation": 500,
    }
    rows = [
        {
            "run_name": "safe",
            "track1_valid": True,
            "non_person_rows": 100,
            "person_false_merge_rate": 0.052,
            "person_purity_mean": 0.968,
            "person_fragmentation": 420,
            "track1_rows": 1000,
        },
        {
            "run_name": "unsafe",
            "track1_valid": True,
            "non_person_rows": 100,
            "person_false_merge_rate": 0.09,
            "person_purity_mean": 0.94,
            "person_fragmentation": 350,
            "track1_rows": 900,
        },
    ]
    selected = select_variant(rows, baseline, {"max_allowed_false_merge_rate_delta": 0.01, "max_allowed_purity_drop": 0.01, "min_person_fragmentation_reduction_for_clear_gain": 60})
    assert selected["selected_variant"] == "safe"
    assert selected["verdict"] == "mlp_association_improves_over_combined_safe_080"
