from deep_oc_sort_3d.person_association.person_merge_policy import apply_person_merge_mapping
from deep_oc_sort_3d.person_reid_association.reid_merge_policy import build_reid_person_merge_mapping


def test_reid_merge_policy_requires_threshold_and_merges_person_only():
    scored_pairs = [
        {
            "candidate_status": "ok",
            "reid_status": "ok",
            "reid_similarity": "0.90",
            "combined_pair_score": "0.10",
            "min_mean_confidence": "0.9",
            "reid_gt_pair_label": "unknown_gt",
            "track_a": "test|Warehouse_023|0|10",
            "track_b": "test|Warehouse_023|0|11",
        },
        {
            "candidate_status": "ok",
            "reid_status": "ok",
            "reid_similarity": "0.50",
            "combined_pair_score": "0.10",
            "min_mean_confidence": "0.9",
            "reid_gt_pair_label": "unknown_gt",
            "track_a": "test|Warehouse_023|0|20",
            "track_b": "test|Warehouse_023|0|21",
        },
    ]
    rows = [
        {"subset": "test", "scene_name": "Warehouse_023", "class_id": "0", "global_track_id": "10", "frame_id": "0"},
        {"subset": "test", "scene_name": "Warehouse_023", "class_id": "0", "global_track_id": "11", "frame_id": "5"},
        {"subset": "test", "scene_name": "Warehouse_023", "class_id": "1", "global_track_id": "11", "frame_id": "5"},
    ]
    mapping, audit, summary = build_reid_person_merge_mapping(
        scored_pairs,
        rows,
        {"reid_similarity_threshold": 0.85, "max_combined_pair_score": 0.2, "prevent_duplicate_frame_keys": True},
    )
    mapped = apply_person_merge_mapping(rows, mapping)
    assert mapping[("test", "Warehouse_023", "0", "11")] == "10"
    assert mapped[1]["global_track_id"] == "10"
    assert mapped[2]["global_track_id"] == "11"
    assert summary["selected_edges_with_reid"] == 1
    assert len(audit) == 2

