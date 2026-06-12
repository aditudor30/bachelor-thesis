import numpy as np

from deep_oc_sort_3d.data.ground_truth import GroundTruthObject
from deep_oc_sort_3d.learned_association.gt_fragment_matcher import match_fragment_to_gt


def _gt(frame_id, object_id, bbox):
    return GroundTruthObject(
        frame_id=frame_id,
        object_type="Person",
        object_id=object_id,
        location_3d=np.asarray([1.0, 2.0, 0.9]),
        bbox3d_scale=np.asarray([0.5, 0.5, 1.8]),
        bbox3d_rotation=np.asarray([0.0, 0.0, 0.0]),
        visible_bboxes_2d={"Camera_0000": bbox},
    )


def test_gt_matcher_selects_dominant_identity():
    fragment = {
        "fragment_id": "f1",
        "scene_name": "Warehouse_000",
        "camera_id": "Camera_0000",
        "num_observations": 3,
        "mean_confidence": 0.9,
        "_observations": [
            {"frame_id": 0, "bbox_xyxy": [10, 10, 30, 50]},
            {"frame_id": 1, "bbox_xyxy": [11, 10, 31, 50]},
            {"frame_id": 2, "bbox_xyxy": [12, 10, 32, 50]},
        ],
    }
    gt = {
        0: [_gt(0, 7, (10, 10, 30, 50))],
        1: [_gt(1, 7, (11, 10, 31, 50))],
        2: [_gt(2, 8, (12, 10, 32, 50))],
    }
    config = {
        "fragment_source": {"min_fragment_length": 3, "min_mean_confidence": 0.05},
        "gt_matching": {
            "min_iou_for_gt_match": 0.3,
            "min_gt_match_ratio": 0.3,
            "min_gt_purity": 0.6,
        },
    }

    result = match_fragment_to_gt(fragment, gt, config)

    assert result["gt_object_id"] == 7
    assert result["gt_identity_id"] == "Warehouse_000_7"
    assert result["gt_match_count"] == 3
    assert result["gt_purity"] == 2.0 / 3.0
    assert result["valid_for_pairs"] is True
