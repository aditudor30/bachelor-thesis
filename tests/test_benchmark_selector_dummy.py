from deep_oc_sort_3d.local_tracker_benchmark.benchmark_selector import select_local_tracker


def test_selector_prefers_safe_fragmentation_improvement_and_ignores_skipped():
    rows = [
        {
            "tracker_name": "current_local_tracker",
            "status": "ok",
            "mean_track_length": 3.0,
            "median_track_length": 1.0,
            "short_track_ratio_le3": 0.80,
            "local_purity_mean": 0.97,
            "false_merge_suspicion_rate": 0.02,
            "nonperson_short_track_ratio_le3": 0.60,
            "runtime_seconds": 1.0,
        },
        {
            "tracker_name": "bytetrack_style_yolo11m",
            "status": "ok",
            "mean_track_length": 7.0,
            "median_track_length": 4.0,
            "short_track_ratio_le3": 0.35,
            "local_purity_mean": 0.96,
            "false_merge_suspicion_rate": 0.025,
            "nonperson_short_track_ratio_le3": 0.62,
            "runtime_seconds": 2.0,
        },
        {
            "tracker_name": "botsort_sbs_mot17_yolo11m",
            "status": "skipped",
            "reason": "optional dependency unavailable",
        },
    ]

    selected = select_local_tracker(rows)

    assert selected["selected_tracker"] == "bytetrack_style_yolo11m"
    assert selected["verdict"] == "bytetrack_style_candidate_for_full_rerun"


def test_selector_keeps_current_tracker_when_candidate_degrades_purity():
    rows = [
        {
            "tracker_name": "current_local_tracker",
            "status": "ok",
            "mean_track_length": 3.0,
            "median_track_length": 1.0,
            "short_track_ratio_le3": 0.8,
            "local_purity_mean": 0.98,
            "false_merge_suspicion_rate": 0.01,
            "nonperson_short_track_ratio_le3": 0.6,
        },
        {
            "tracker_name": "botsort_style_no_reid_yolo11m",
            "status": "ok",
            "mean_track_length": 10.0,
            "median_track_length": 5.0,
            "short_track_ratio_le3": 0.2,
            "local_purity_mean": 0.80,
            "false_merge_suspicion_rate": 0.15,
            "nonperson_short_track_ratio_le3": 0.7,
        },
    ]

    selected = select_local_tracker(rows)

    assert selected["selected_tracker"] == "current_local_tracker"
    assert selected["verdict"] == "current_tracker_still_best"
