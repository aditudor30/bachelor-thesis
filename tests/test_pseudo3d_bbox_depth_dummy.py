import numpy as np

from deep_oc_sort_3d.pseudo3d.bbox_depth_estimator import estimate_depth_from_bbox_height
from deep_oc_sort_3d.pseudo3d.camera_model_adapter import CameraModel


def _camera():
    return CameraModel(np.asarray([[100.0, 0.0, 50.0], [0.0, 200.0, 60.0], [0.0, 0.0, 1.0]]), None, None, 100, 100)


def test_bbox_height_depth_estimates_depth():
    depth, error = estimate_depth_from_bbox_height((0.0, 0.0, 10.0, 20.0), 2.0, _camera(), {"min_bbox_height_px": 8})

    assert error is None
    assert depth == 20.0


def test_bbox_height_depth_rejects_tiny_bbox():
    depth, error = estimate_depth_from_bbox_height((0.0, 0.0, 10.0, 2.0), 2.0, _camera(), {"min_bbox_height_px": 8})

    assert depth is None
    assert error == "bbox_height_too_small"


def test_bbox_height_depth_clamps_depth():
    depth, error = estimate_depth_from_bbox_height(
        (0.0, 0.0, 10.0, 1.0),
        2.0,
        _camera(),
        {"min_bbox_height_px": 1, "max_depth_m": 50.0, "min_depth_m": 0.1},
    )

    assert error is None
    assert depth == 50.0

