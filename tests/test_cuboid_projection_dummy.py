import numpy as np

from deep_oc_sort_3d.visualization3d.cuboid_projection import (
    is_projected_cuboid_visible,
    project_cuboid_to_image,
    project_points_to_image,
)


def test_project_points_to_image_with_dummy_camera_matrix():
    calibration = {
        "camera_matrix": np.asarray(
            [
                [100.0, 0.0, 50.0, 0.0],
                [0.0, 100.0, 50.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
            ],
            dtype=float,
        )
    }
    points = np.asarray([[0.0, 0.0, 10.0], [1.0, 1.0, 10.0]], dtype=float)
    projected = project_points_to_image(points, calibration)
    assert projected.shape == (2, 2)
    np.testing.assert_allclose(projected[0], np.asarray([50.0, 50.0]))
    assert is_projected_cuboid_visible(projected, 100, 100)


def test_project_cuboid_to_image_handles_invalid_inputs():
    result = project_cuboid_to_image([0, 0, 10], [1, 0, 1], 0.0, {})
    assert result["success"] is False
    assert result["points_2d"] is None

