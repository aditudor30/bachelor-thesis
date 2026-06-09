from deep_oc_sort_3d.reid_finetuned_association.finetuned_pair_scorer import normalize_pair_score_rows, reid_score_distribution, threshold_summary


def test_pair_scorer_threshold_columns():
    rows = [
        {
            "subset": "test",
            "scene_name": "Warehouse_023",
            "class_id": 0,
            "track_a": "a",
            "track_b": "b",
            "global_track_id_a": "1",
            "global_track_id_b": "2",
            "cameras_a": "Camera_0000",
            "cameras_b": "Camera_0001",
            "end_a": 10,
            "start_b": 12,
            "temporal_gap": 2,
            "entry_exit_distance_3d": 1.0,
            "velocity_score": 0.2,
            "geometry_pair_score": 0.1,
            "combined_pair_score": 0.2,
            "reid_similarity": 0.76,
            "reject_reason": "ok",
            "reid_status": "ok",
        }
    ]
    config = {"sweep": {"thresholds": [0.65, 0.75, 0.80]}}
    normalized = normalize_pair_score_rows(rows, config)
    assert normalized[0]["passes_threshold_065"] == "1"
    assert normalized[0]["passes_threshold_075"] == "1"
    assert normalized[0]["passes_threshold_080"] == "0"
    summary = threshold_summary(normalized, [0.65, 0.75, 0.80])
    assert summary[0]["passing_pairs"] == 1
    distribution = reid_score_distribution(normalized)
    assert distribution["num_pairs_with_reid"] == 1
    assert distribution["median"] == 0.76
