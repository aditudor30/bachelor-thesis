from deep_oc_sort_3d.person_cleanup.person_cleanup_selection import select_best_person_cleanup


def test_person_cleanup_selection_prefers_clean_non_person_preserving_run():
    v2 = {
        "track1_rows": 1000,
        "person_rows": 700,
        "non_person_rows": 300,
        "person_fragmentation_approx": 100,
        "global_purity_mean": 0.97,
        "false_merge_rate": 0.05,
    }
    runs = [
        {
            "run_name": "bad_non_person",
            "track1_rows": 850,
            "person_rows": 650,
            "non_person_rows": 200,
            "person_fragmentation_approx": 90,
            "global_purity_mean": 0.97,
            "false_merge_rate": 0.05,
            "track1_validation_errors": 0,
        },
        {
            "run_name": "safe",
            "track1_rows": 930,
            "person_rows": 630,
            "non_person_rows": 300,
            "person_fragmentation_approx": 85,
            "global_purity_mean": 0.969,
            "false_merge_rate": 0.052,
            "track1_validation_errors": 0,
        },
    ]

    result = select_best_person_cleanup(runs, v2)

    assert result["best_run"] == "safe"
    assert result["verdict"] == "person_cleanup_ready_for_submission_candidate"

