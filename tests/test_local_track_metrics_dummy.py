from deep_oc_sort_3d.local_tracker_benchmark.local_track_metrics import compute_track_metrics


def _row(track_id, frame_id, confidence=0.8):
    return {
        "subset": "official_val",
        "scene_name": "Warehouse_020",
        "camera_id": "Camera_0000",
        "class_id": 0,
        "track_id": track_id,
        "frame_id": frame_id,
        "confidence": confidence,
    }


def test_local_track_length_metrics_are_computed_per_track():
    rows = [
        _row(1, 0),
        _row(1, 1),
        _row(1, 2),
        _row(2, 5),
    ]

    metrics = compute_track_metrics(rows)

    assert metrics["num_tracks"] == 2
    assert metrics["num_records"] == 4
    assert metrics["mean_track_length"] == 2.0
    assert metrics["median_track_length"] == 2.0
    assert metrics["num_length_1_tracks"] == 1
    assert metrics["short_track_ratio_le3"] == 1.0
    assert metrics["track_duration_mean"] == 2.0
