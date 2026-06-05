from deep_oc_sort_3d.person_association.person_merge_policy import (
    apply_person_merge_mapping,
    build_person_merge_mapping,
)


def test_person_merge_policy_merges_person_only():
    scored_pairs = [
        {
            "candidate_status": "ok",
            "score_status": "scored",
            "pair_score": "0.1",
            "min_mean_confidence": "0.9",
            "same_gt_diagnostic": "unknown_gt",
            "track_a": "test|Warehouse_023|0|10",
            "track_b": "test|Warehouse_023|0|11",
        }
    ]
    rows = [
        {"subset": "test", "scene_name": "Warehouse_023", "class_id": "0", "global_track_id": "10", "frame_id": "0"},
        {"subset": "test", "scene_name": "Warehouse_023", "class_id": "0", "global_track_id": "11", "frame_id": "5"},
        {"subset": "test", "scene_name": "Warehouse_023", "class_id": "1", "global_track_id": "11", "frame_id": "5"},
    ]
    mapping, audit = build_person_merge_mapping(
        scored_pairs,
        rows,
        {"max_pair_score": 0.2, "min_mean_confidence": 0.1, "prevent_duplicate_frame_keys": True},
    )
    mapped = apply_person_merge_mapping(rows, mapping)
    assert mapping[("test", "Warehouse_023", "0", "11")] == "10"
    assert mapped[1]["global_track_id"] == "10"
    assert mapped[2]["global_track_id"] == "11"
    assert len(audit) == 1

