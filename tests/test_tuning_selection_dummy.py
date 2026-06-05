from deep_oc_sort_3d.global_tuning.tuning_selection import select_best_run


def test_tuning_selection_prefers_valid_fragmentation_reduction():
    v2 = {
        "fragmentation_approx": 100.0,
        "global_purity_mean": 0.975,
        "false_merge_rate": 0.05,
        "track1_rows": 1000,
    }
    runs = [
        {
            "run_name": "bad_false_merge",
            "fragmentation_approx": 80.0,
            "global_purity_mean": 0.974,
            "false_merge_rate": 0.09,
            "track1_rows": 900,
            "track1_validation_errors": 0,
        },
        {
            "run_name": "balanced",
            "fragmentation_approx": 85.0,
            "global_purity_mean": 0.973,
            "false_merge_rate": 0.055,
            "track1_rows": 950,
            "track1_validation_errors": 0,
        },
    ]

    result = select_best_run(runs, v2)

    assert result["best_run"] == "balanced"
    assert result["verdict"] == "best_run_ready_for_submission_candidate"

