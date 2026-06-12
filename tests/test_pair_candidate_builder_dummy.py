import random

from deep_oc_sort_3d.learned_association.pair_candidate_builder import build_scene_candidate_pairs


def _fragment(fragment_id, identity, camera):
    return {
        "fragment_id": fragment_id,
        "gt_identity_id": identity,
        "camera_id": camera,
        "frame_start": 0,
        "frame_end": 5,
        "scene_id": 0,
        "_embedding": None,
    }


def test_candidate_builder_creates_same_and_different_identity_labels():
    fragments = [
        _fragment("a1", "Warehouse_000_1", "Camera_0000"),
        _fragment("a2", "Warehouse_000_1", "Camera_0001"),
        _fragment("b1", "Warehouse_000_2", "Camera_0000"),
        _fragment("b2", "Warehouse_000_2", "Camera_0001"),
    ]
    settings = {
        "max_positive_pairs_per_identity": 10,
        "max_negative_pairs_per_identity": 10,
        "hard_negative_top_k": 2,
        "include_hard_negatives": True,
    }

    pairs = build_scene_candidate_pairs(
        fragments, "train", "Warehouse_000", settings, random.Random(42)
    )

    assert any(row["same_identity"] == 1 for row in pairs)
    assert any(row["same_identity"] == 0 for row in pairs)
    for row in pairs:
        expected = int(row["gt_identity_a"] == row["gt_identity_b"])
        assert row["same_identity"] == expected
