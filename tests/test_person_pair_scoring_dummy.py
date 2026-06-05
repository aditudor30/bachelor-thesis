from deep_oc_sort_3d.person_association.person_pair_scoring import score_person_pair


def test_person_pair_scoring_prefers_closer_pair():
    base = {
        "candidate_status": "ok",
        "temporal_gap": 3,
        "entry_exit_distance_3d": 0.2,
        "expected_position_error": 0.2,
        "velocity_angle": 5,
        "min_mean_confidence": 0.8,
    }
    far = dict(base)
    far["entry_exit_distance_3d"] = 2.0
    near_score = score_person_pair(base, {"max_entry_exit_distance": 2.0})["pair_score"]
    far_score = score_person_pair(far, {"max_entry_exit_distance": 2.0})["pair_score"]
    assert near_score < far_score

