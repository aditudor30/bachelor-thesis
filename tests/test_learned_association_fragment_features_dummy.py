import pytest

from deep_oc_sort_3d.learned_association.fragment_feature_builder import normalize_fragment_record


def test_fragment_features_are_aggregated_from_sampled_trajectories():
    record = {
        "candidate_id": "candidate_1",
        "class_id": 0,
        "class_name": "Person",
        "camera_id": "Camera_0000",
        "mean_confidence": 0.8,
        "trajectory_2d_sampled": [[0, 10, 20, 30, 60], [2, 12, 20, 32, 60]],
        "trajectory_3d_sampled": [[0, 1.0, 2.0, 0.9], [2, 3.0, 2.0, 0.9]],
    }

    fragment = normalize_fragment_record(record, "motion_clean_candidates", "train", "Warehouse_000")

    assert fragment["fragment_id"] == "candidate_1"
    assert fragment["split"] == "train"
    assert fragment["num_observations"] == 2
    assert fragment["bbox_width_mean"] == pytest.approx(20.0)
    assert fragment["bbox_height_mean"] == pytest.approx(40.0)
    assert fragment["center_x_mean"] == pytest.approx(2.0)
    assert fragment["velocity_x"] == pytest.approx(1.0)
