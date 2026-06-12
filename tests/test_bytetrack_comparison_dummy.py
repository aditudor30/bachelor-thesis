from deep_oc_sort_3d.bytetrack_local_pipeline.bytetrack_comparison import (
    decide_final_verdict,
    metric_deltas,
)


def test_comparison_deltas_and_submission_candidate_verdict():
    current = {
        "local_tracking": {"approx_fragmentation": 1000, "median_track_length": 6, "num_records": 1000},
        "global_association": {"fragmentation_approx": 500, "global_purity_mean": 0.97, "false_merge_rate": 0.05},
        "track1": {"rows": 100, "validation_errors": 0},
    }
    candidate = {
        "local_tracking": {"approx_fragmentation": 200, "median_track_length": 50, "num_records": 900},
        "global_association": {"fragmentation_approx": 400, "global_purity_mean": 0.971, "false_merge_rate": 0.052},
        "track1": {"rows": 110, "validation_errors": 0},
    }
    deltas = metric_deltas(current, candidate)
    verdict = decide_final_verdict(candidate, current, deltas, {"selection": {}})

    assert deltas["local_fragmentation_delta"] == -800.0
    assert deltas["global_fragmentation_delta"] == -100.0
    assert verdict["label"] == "baseline_v2_bytetrack_local_ready_for_submission_candidate"


def test_comparison_rejects_invalid_track1():
    candidate = {"track1": {"validation_errors": 2}}
    verdict = decide_final_verdict(candidate, {}, {}, {"selection": {}})
    assert verdict["label"] == "baseline_v2_bytetrack_local_invalid_fix_required"
