"""Dummy tests for transition summary."""

from deep_oc_sort_3d.mtmc.transition_diagnostics import compute_transition_pair_metrics, summarize_transition_pairs
from tests.test_transition_diagnostics_dummy import make_transition_candidate


def test_transition_summary_counts_labels_and_classes():
    a = make_transition_candidate("a", "Camera_0000", 0, 10, 0.0, 1.0, gt_id=1)
    b = make_transition_candidate("b", "Camera_0001", 20, 30, 1.2, 2.0, gt_id=1)
    c = make_transition_candidate("c", "Camera_0002", 20, 30, 1.2, 2.0, gt_id=2)
    d = make_transition_candidate("d", "Camera_0003", 20, 30, 1.2, 2.0, gt_id=None)
    pairs = [
        compute_transition_pair_metrics(a, b, {"max_expected_position_error": 20.0}),
        compute_transition_pair_metrics(a, c, {"max_expected_position_error": 20.0}),
        compute_transition_pair_metrics(a, d, {"max_expected_position_error": 20.0}),
    ]
    summary = summarize_transition_pairs(pairs)
    assert summary["total_pairs"] == 3
    assert summary["true_transition"] == 1
    assert summary["false_transition"] == 1
    assert summary["unknown_gt"] == 1
    assert summary["per_class_counts"]["Person"] == 3
