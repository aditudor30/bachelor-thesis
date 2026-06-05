from deep_oc_sort_3d.person_reid_association.reid_pair_scoring import score_reid_person_pair, summarize_reid_pair_scores


def _row(similarity):
    return {
        "candidate_status": "ok",
        "temporal_gap": "1",
        "entry_exit_distance_3d": "0.2",
        "expected_position_error": "0.2",
        "velocity_angle": "10",
        "min_mean_confidence": "0.9",
        "reid_status": "ok",
        "reid_similarity": str(similarity),
    }


def test_reid_pair_scoring_prefers_higher_similarity():
    high = score_reid_person_pair(_row(0.9), {"reid_weight": 0.5})
    low = score_reid_person_pair(_row(0.4), {"reid_weight": 0.5})
    assert high["combined_pair_score"] < low["combined_pair_score"]


def test_reid_pair_score_summary_counts_threshold_passes():
    rows = [score_reid_person_pair(_row(0.86)), score_reid_person_pair(_row(0.79))]
    summary = summarize_reid_pair_scores(rows, threshold=0.85)
    assert summary["pairs_with_reid"] == 2
    assert summary["pairs_passing_reid_threshold"] == 1

