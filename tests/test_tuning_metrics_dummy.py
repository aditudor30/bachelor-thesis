from deep_oc_sort_3d.global_tuning.tuning_metrics import compute_metric_deltas


def test_tuning_metrics_deltas_compute_fragmentation_reduction():
    baseline = {
        "fragmentation_approx": 100.0,
        "false_merge_rate": 0.05,
        "global_purity_mean": 0.97,
        "track1_rows": 1000,
    }
    run = {
        "fragmentation_approx": 80.0,
        "false_merge_rate": 0.06,
        "global_purity_mean": 0.969,
        "track1_rows": 900,
    }

    deltas = compute_metric_deltas(run, baseline, "vs_v2")

    assert deltas["vs_v2_fragmentation_approx_delta"] == -20.0
    assert abs(deltas["vs_v2_fragmentation_reduction"] - 0.2) < 1e-9
    assert abs(deltas["vs_v2_false_merge_rate_delta"] - 0.01) < 1e-9

