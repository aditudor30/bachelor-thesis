import numpy as np

from deep_oc_sort_3d.visualization3d.cuboid_geometry import (
    compute_3d_cuboid_corners,
    get_cuboid_edges,
    validate_cuboid_inputs,
)


def test_compute_3d_cuboid_corners_shape_and_yaw_zero():
    center = np.asarray([0.0, 0.0, 1.0], dtype=float)
    dimensions = np.asarray([2.0, 4.0, 6.0], dtype=float)
    corners = compute_3d_cuboid_corners(center, dimensions, 0.0)
    assert corners.shape == (8, 3)
    np.testing.assert_allclose(corners[0], np.asarray([-1.0, -2.0, -2.0]))
    np.testing.assert_allclose(corners[6], np.asarray([1.0, 2.0, 4.0]))
    assert len(get_cuboid_edges()) == 12


def test_validate_cuboid_inputs_rejects_non_positive_dimensions():
    assert validate_cuboid_inputs([0, 0, 0], [1, 1, 1], 0.0)
    assert not validate_cuboid_inputs([0, 0, 0], [1, 0, 1], 0.0)

