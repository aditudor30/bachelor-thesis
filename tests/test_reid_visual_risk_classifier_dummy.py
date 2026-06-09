from deep_oc_sort_3d.reid_visual_decision.visual_risk_classifier import classify_merge_event


def test_reid_visual_risk_classifier_labels_good_and_low_evidence():
    config = {"heuristics": {"reid_similarity_threshold": 0.80, "high_similarity": 0.86, "min_crops_per_fragment": 2}}
    event = {"reid_similarity": "0.90", "spatial_distance": "2.0", "temporal_gap": "10"}
    evidence = {"num_crops_a": 3, "num_crops_b": 3}
    assert classify_merge_event(event, evidence, config)["auto_label"] == "likely_good"
    evidence = {"num_crops_a": 1, "num_crops_b": 3}
    assert classify_merge_event(event, evidence, config)["auto_label"] == "not_enough_visual_evidence"

