from deep_oc_sort_3d.person_reid_association.reid_association_selection import select_best_reid_association


def test_reid_association_selection_rejects_noop_and_selects_safe_gain():
    baseline = {
        "person_fragmentation_approx": 100,
        "global_purity_mean": 0.95,
        "false_merge_rate": 0.05,
        "non_person_rows": 10,
    }
    runs = [
        {
            "run_name": "diagnostic_only",
            "run_status": "ok",
            "merges_applied": 0,
            "track1_validation_errors": 0,
            "person_fragmentation_approx": 100,
            "global_purity_mean": 0.95,
            "false_merge_rate": 0.05,
            "non_person_rows": 10,
        },
        {
            "run_name": "reid_strict",
            "run_status": "ok",
            "merges_applied": 5,
            "track1_validation_errors": 0,
            "person_fragmentation_approx": 90,
            "global_purity_mean": 0.949,
            "false_merge_rate": 0.052,
            "non_person_rows": 10,
        },
    ]
    recommendation = select_best_reid_association(runs, baseline)
    assert recommendation["best_run"] == "reid_strict"
    assert recommendation["verdict"] == "reid_association_improves_v2"
