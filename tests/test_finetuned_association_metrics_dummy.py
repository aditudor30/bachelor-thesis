from deep_oc_sort_3d.reid_finetuned_association.finetuned_association_metrics import metric_delta, sweep_summary_row


def test_metric_delta_numeric():
    run = {"person_rows": 90}
    baseline = {"person_rows": 100}
    assert metric_delta(run, baseline, "person_rows") == -10.0


def test_sweep_summary_row_keeps_non_person_delta():
    metrics = {
        "run_name": "threshold_075",
        "run_status": "ok",
        "track1_validation_errors": 0,
        "track1_validation_status": "ok",
        "track1_rows": 10,
        "global_unique_tracks": 5,
        "person_unique_tracks": 3,
        "multi_camera_tracks": 2,
        "person_singleton_tracks": 1,
        "global_purity_mean": 0.98,
        "person_purity": 0.99,
        "false_merge_rate": 0.01,
        "person_false_merge_rate": 0.02,
        "fragmentation_approx": 4,
        "person_fragmentation_approx": 3,
        "vs_v2_person_fragmentation_approx_delta": -1,
        "vs_v2_track1_rows_delta": 0,
        "vs_v2_person_rows_delta": -2,
        "vs_v2_non_person_rows_delta": 0,
    }
    row = sweep_summary_row(metrics)
    assert row["track1_valid"] == "1"
    assert row["non_person_rows_delta"] == 0
    assert row["person_fragmentation_delta"] == -1
