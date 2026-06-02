"""Dummy tests for final export diagnostic evaluation."""

from deep_oc_sort_3d.final_export.export_eval import evaluate_global_frame_records
from tests.test_generic_export_dummy import make_global_record


def test_final_export_eval_purity_false_merge_fragmentation():
    a = make_global_record(0, 1)
    a.matched_gt_object_id = 10
    b = make_global_record(1, 1)
    b.matched_gt_object_id = 10
    c = make_global_record(2, 1)
    c.matched_gt_object_id = 11
    d = make_global_record(3, 2)
    d.matched_gt_object_id = 10
    metrics = evaluate_global_frame_records([a, b, c, d])
    assert metrics["num_records"] == 4
    assert metrics["false_merge_count"] == 1
    assert metrics["fragmentation_approx"] == 1
    assert metrics["global_id_purity_mean"] < 1.0
