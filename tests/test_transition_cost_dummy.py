"""Dummy tests for transition association cost."""

from deep_oc_sort_3d.mtmc.transition_cost import compute_transition_cost
from deep_oc_sort_3d.mtmc.transition_diagnostics import compute_transition_pair_metrics
from tests.test_transition_diagnostics_dummy import make_transition_candidate


def test_good_transition_cost_is_accepted():
    a = make_transition_candidate("a", "Camera_0000", 0, 10, 0.0, 1.0, gt_id=1)
    b = make_transition_candidate("b", "Camera_0001", 20, 30, 1.2, 2.0, gt_id=1)
    pair = compute_transition_pair_metrics(a, b, {"max_expected_position_error": 20.0})
    cost, accepted, reason = compute_transition_cost(pair, {"max_expected_position_error": 20.0})
    assert cost >= 0.0
    assert accepted
    assert reason == "ok"


def test_large_gap_and_large_distance_are_rejected():
    a = make_transition_candidate("a", "Camera_0000", 0, 10, 0.0, 1.0, gt_id=1)
    far_time = make_transition_candidate("b", "Camera_0001", 500, 510, 1.2, 2.0, gt_id=1)
    far_space = make_transition_candidate("c", "Camera_0002", 20, 30, 100.0, 101.0, gt_id=1)
    pair_gap = compute_transition_pair_metrics(a, far_time, {"max_temporal_gap": 90})
    pair_space = compute_transition_pair_metrics(a, far_space, {"max_entry_exit_distance": 4.0})
    assert compute_transition_cost(pair_gap, {"max_temporal_gap": 90})[2] == "temporal_gap_too_large"
    assert compute_transition_cost(pair_space, {"max_entry_exit_distance": 4.0})[2] == "entry_exit_distance_too_large"
