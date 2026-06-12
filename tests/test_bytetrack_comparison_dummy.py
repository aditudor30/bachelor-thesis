from deep_oc_sort_3d.bytetrack_local_pipeline.bytetrack_comparison import (
    collect_local_tracking_metrics,
    decide_final_verdict,
    is_local_track_csv,
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


def test_comparison_rejects_large_coverage_loss():
    current = {
        "local_tracking": {"num_records": 1000, "gt_matched_records": 500},
        "global_association": {"multi_camera_tracks": 100},
        "track1": {"rows": 1000, "validation_errors": 0},
    }
    candidate = {
        "local_tracking": {"num_records": 300, "gt_matched_records": 200},
        "global_association": {"multi_camera_tracks": 20},
        "track1": {"rows": 250, "validation_errors": 0},
    }

    verdict = decide_final_verdict(candidate, current, {}, {"selection": {}})

    assert verdict["label"] == "baseline_v2_bytetrack_local_valid_but_coverage_too_low"
    assert "track1_row_retention_below_minimum" in verdict["reasons"]
    assert "multi_camera_track_retention_below_minimum" in verdict["reasons"]


def test_local_tracking_metrics_ignore_auxiliary_csv(tmp_path):
    track_path = tmp_path / "official_val" / "Warehouse_020" / "Camera_0000.csv"
    track_path.parent.mkdir(parents=True)
    track_path.write_text(
        "scene_id,scene_name,split,camera_id,frame_id,local_track_id,detection_id,"
        "class_id,class_name,confidence,x1,y1,x2,y2,w,h,center_x,center_y,center_z,"
        "width_3d,length_3d,height_3d,yaw,matched_gt_object_id,matched_gt,track_age,"
        "track_hits,track_misses,track_state\n"
        "20,Warehouse_020,val,Camera_0000,0,1,1,0,Person,0.9,10,20,30,60,20,40,"
        "0,0,0,1,1,1,0,7,1,1,1,0,confirmed\n",
        encoding="utf-8",
    )
    auxiliary_path = tmp_path / "eval" / "local_tracking_eval.csv"
    auxiliary_path.parent.mkdir(parents=True)
    auxiliary_path.write_text(
        "num_tracks,mean_track_length,approx_fragmentation\n10,5.0,2\n",
        encoding="utf-8",
    )

    assert is_local_track_csv(track_path) is True
    assert is_local_track_csv(auxiliary_path) is False

    summary = collect_local_tracking_metrics(tmp_path)
    assert summary["files_read"] == 1
    assert summary["skipped_non_track_csv_files"] == 1
    assert summary["num_records"] == 1
