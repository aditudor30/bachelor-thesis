from deep_oc_sort_3d.learned_association_application.candidate_pair_scorer import normalize_scored_pair


def test_candidate_pair_scorer_sets_threshold_gates():
    row = {
        "class_id": 0,
        "class_name": "Person",
        "reid_similarity": 0.84,
        "same_camera_temporal_conflict": 0,
        "temporal_overlap_conflict": 0,
        "large_spatial_gap_flag": 0,
        "large_temporal_gap_flag": 0,
    }
    scored = normalize_scored_pair(row, 0.80)
    assert scored["passes_mlp_077"] == 1
    assert scored["passes_mlp_085"] == 0
    assert scored["passes_reid_080"] == 1
    assert scored["valid_for_merge"] == 1
