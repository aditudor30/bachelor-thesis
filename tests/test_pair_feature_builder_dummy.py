import numpy as np
import pytest

from deep_oc_sort_3d.learned_association.pair_feature_builder import build_pair_features


def _fragment(fragment_id, camera, start, end, center_start, center_end, embedding):
    center_mean = [(a + b) / 2.0 for a, b in zip(center_start, center_end)]
    return {
        "fragment_id": fragment_id,
        "camera_id": camera,
        "frame_start": start,
        "frame_end": end,
        "duration": end - start + 1,
        "num_observations": end - start + 1,
        "mean_confidence": 0.9,
        "min_confidence": 0.7,
        "gt_purity": 1.0,
        "bbox_area_mean": 400.0,
        "bbox_height_mean": 40.0,
        "fragment_quality": "good",
        "center_x_start": center_start[0],
        "center_y_start": center_start[1],
        "center_z_start": center_start[2],
        "center_x_end": center_end[0],
        "center_y_end": center_end[1],
        "center_z_end": center_end[2],
        "center_x_mean": center_mean[0],
        "center_y_mean": center_mean[1],
        "center_z_mean": center_mean[2],
        "velocity_x": 1.0,
        "velocity_y": 0.0,
        "velocity_z": 0.0,
        "speed_mean": 1.0,
        "embedding_available": True,
        "_embedding": np.asarray(embedding, dtype=np.float32),
    }


def test_pair_features_cover_temporal_camera_reid_geometry_and_conflicts():
    fragment_a = _fragment("a", "Camera_0000", 0, 4, [0, 0, 0], [4, 0, 0], [1, 0])
    fragment_b = _fragment("b", "Camera_0001", 7, 10, [7, 0, 0], [10, 0, 0], [1, 0])
    candidate = {
        "pair_id": "pair",
        "same_identity": 1,
        "split": "train",
        "scene_name": "Warehouse_000",
        "_fragment_a": fragment_a,
        "_fragment_b": fragment_b,
    }

    row = build_pair_features(candidate, {"Camera_0000__Camera_0001"})

    assert row["reid_similarity"] == pytest.approx(1.0)
    assert row["temporal_gap"] == 2
    assert row["temporal_overlap"] == 0
    assert row["temporal_order"] == "a_before_b"
    assert row["camera_pair"] == "Camera_0000__Camera_0001"
    assert row["cross_camera"] == 1
    assert row["min_endpoint_distance_3d"] == pytest.approx(3.0)
    assert row["missing_reid_flag"] == 0
    assert row["missing_geometry_flag"] == 0


def test_same_camera_overlap_sets_conflict_flag():
    fragment_a = _fragment("a", "Camera_0000", 0, 5, [0, 0, 0], [5, 0, 0], [1, 0])
    fragment_b = _fragment("b", "Camera_0000", 3, 8, [3, 0, 0], [8, 0, 0], [0, 1])
    row = build_pair_features(
        {"pair_id": "p", "same_identity": 0, "_fragment_a": fragment_a, "_fragment_b": fragment_b}
    )

    assert row["temporal_overlap"] == 3
    assert row["same_camera_temporal_conflict"] == 1
    assert row["reid_similarity"] == pytest.approx(0.0)
