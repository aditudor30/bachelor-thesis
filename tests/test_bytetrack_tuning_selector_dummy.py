from deep_oc_sort_3d.bytetrack_tuning.tuning_selector import select_tuned_variant


def _run(local_records, gt_records, track1_rows, multi_camera, purity=0.97, false_merge=0.05):
    return {
        "status": "ok",
        "local_tracking": {
            "num_records": local_records,
            "gt_matched_records": gt_records,
            "median_track_length": 20,
            "short_track_ratio_le3": 0.1,
            "approx_fragmentation": 50,
            "approx_id_switches": 20,
        },
        "global_association": {
            "global_tracks": 80,
            "multi_camera_tracks": multi_camera,
            "global_purity_mean": purity,
            "false_merge_rate": false_merge,
            "fragmentation_approx": 40,
        },
        "track1": {"rows": track1_rows, "validation_errors": 0},
    }


def test_selector_enforces_hard_coverage_gates():
    baseline = _run(100, 90, 100, 50)
    good = _run(90, 85, 80, 35)
    good["local_tracking"]["median_track_length"] = 30
    good["local_tracking"]["short_track_ratio_le3"] = 0.05
    good["global_association"]["fragmentation_approx"] = 30
    metrics = {
        "baselines": {"baseline_v2_current": baseline},
        "variants": {
            "low": _run(50, 85, 30, 10),
            "good": good,
        },
    }
    config = {
        "selection": {
            "min_local_records_retention": 0.85,
            "min_gt_matched_retention": 0.90,
            "min_track1_rows_retention": 0.75,
            "min_multi_camera_tracks_retention": 0.60,
            "max_allowed_purity_drop": 0.01,
            "max_allowed_false_merge_rate_delta": 0.01,
            "require_track1_errors_zero": True,
        }
    }
    result = select_tuned_variant(metrics, config)
    assert result["selected_variant"] == "good"
    assert result["verdict"]["label"] == "bytetrack_tuned_ready_for_full_submission_candidate"


def test_selector_handles_skipped_variants():
    baseline = _run(100, 90, 100, 50)
    metrics = {
        "baselines": {"baseline_v2_current": baseline},
        "variants": {"phase_a_only": {"status": "phase_a_only", "local_tracking": {"num_records": 95}}},
    }
    result = select_tuned_variant(metrics, {"selection": {}})
    assert result["selected_variant"] is None
    assert result["verdict"]["label"] == "bytetrack_tuning_invalid_fix_required"
