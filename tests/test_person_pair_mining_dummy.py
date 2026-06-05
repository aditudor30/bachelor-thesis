from deep_oc_sort_3d.person_association.person_pair_mining import (
    fragment_from_rows,
    mine_person_candidate_pairs,
)


def test_person_pair_mining_finds_close_non_overlapping_pair():
    rows_a = [
        {
            "subset": "official_val",
            "scene_name": "Warehouse_020",
            "class_id": "0",
            "class_name": "Person",
            "global_track_id": "10",
            "camera_id": "Camera_0000",
            "frame_id": "0",
            "center_x": "0.0",
            "center_y": "0.0",
            "center_z": "0.0",
            "confidence": "0.8",
            "matched_gt_object_id": "42",
        },
        {
            "subset": "official_val",
            "scene_name": "Warehouse_020",
            "class_id": "0",
            "class_name": "Person",
            "global_track_id": "10",
            "camera_id": "Camera_0000",
            "frame_id": "5",
            "center_x": "0.5",
            "center_y": "0.0",
            "center_z": "0.0",
            "confidence": "0.8",
            "matched_gt_object_id": "42",
        },
    ]
    rows_b = [
        {
            "subset": "official_val",
            "scene_name": "Warehouse_020",
            "class_id": "0",
            "class_name": "Person",
            "global_track_id": "11",
            "camera_id": "Camera_0001",
            "frame_id": "8",
            "center_x": "0.8",
            "center_y": "0.0",
            "center_z": "0.0",
            "confidence": "0.7",
            "matched_gt_object_id": "42",
        }
    ]
    frag_a = fragment_from_rows(("official_val", "Warehouse_020", "0", "10"), rows_a)
    frag_b = fragment_from_rows(("official_val", "Warehouse_020", "0", "11"), rows_b)
    pairs = mine_person_candidate_pairs(
        [frag_a, frag_b],
        {
            "include_rejected": True,
            "allow_same_camera": False,
            "max_temporal_gap": 10,
            "max_entry_exit_distance": 1.0,
            "require_no_temporal_conflict": True,
        },
        show_progress=False,
    )
    assert len(pairs) == 1
    assert pairs[0]["candidate_status"] == "ok"
    assert pairs[0]["same_gt_diagnostic"] == "true_match"

