import numpy as np
import pytest

from deep_oc_sort_3d.pseudo3d.stabilized_yaw import estimate_yaw_from_smoothed_motion


def test_yaw_default_for_small_motion() -> None:
    centers = np.asarray([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]])
    yaws, sources = estimate_yaw_from_smoothed_motion(centers, [1, 2], min_displacement=0.5, default_yaw=0.25)
    assert yaws[1] == pytest.approx(0.25)
    assert sources[1] == "class_default"


def test_yaw_from_motion() -> None:
    centers = np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    yaws, sources = estimate_yaw_from_smoothed_motion(centers, [1, 2], min_displacement=0.5)
    assert yaws[1] == pytest.approx(0.0)
    assert sources[1] == "motion_direction_smoothed"
