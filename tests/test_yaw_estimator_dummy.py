import numpy as np

from deep_oc_sort_3d.pseudo3d.pseudo3d_types import Pseudo3DOutput
from deep_oc_sort_3d.pseudo3d.yaw_estimator import estimate_default_yaw, estimate_yaw_from_motion


def _output(center):
    return Pseudo3DOutput(
        center_3d=np.asarray(center, dtype=float),
        dimensions_3d=None,
        yaw=None,
        depth=None,
        confidence_3d=1.0,
        center_3d_source="pseudo3d_bbox_height",
        dimensions_3d_source="class_prior",
        yaw_source="unknown",
        depth_source="bbox_height_prior",
        is_gt_derived=False,
        is_estimated_for_test=True,
        pseudo3d_method="bbox_height_depth",
        pseudo3d_version="0.1",
    )


def test_yaw_default_and_motion_direction():
    assert estimate_default_yaw({"class_default_yaw": 0.3}) == 0.3

    yaw = estimate_yaw_from_motion([_output([0.0, 0.0, 0.0]), _output([1.0, 0.0, 0.0])], {"min_track_displacement_for_yaw": 0.5})

    assert yaw == 0.0

