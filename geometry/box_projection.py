"""3D box corner creation and image projection helpers."""

from typing import Optional

import numpy as np

from deep_oc_sort_3d.data.calibration import CameraCalibration
from deep_oc_sort_3d.geometry.camera_geometry import project_world_to_image


def yaw_to_rotation_matrix(yaw: float) -> np.ndarray:
    """Return a rotation matrix around the vertical z axis."""
    cos_yaw = float(np.cos(yaw))
    sin_yaw = float(np.sin(yaw))
    return np.asarray(
        [
            [cos_yaw, -sin_yaw, 0.0],
            [sin_yaw, cos_yaw, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )


def create_3d_box_corners(
    center: np.ndarray,
    dimensions: np.ndarray,
    yaw: float,
) -> np.ndarray:
    """Create 8 world-space box corners from center, dimensions, and yaw."""
    center_arr = np.asarray(center, dtype=float).reshape(3)
    dims = np.asarray(dimensions, dtype=float).reshape(3)
    width, length, height = dims
    x = width * 0.5
    y = length * 0.5
    z = height * 0.5
    local = np.asarray(
        [
            [-x, -y, -z],
            [x, -y, -z],
            [x, y, -z],
            [-x, y, -z],
            [-x, -y, z],
            [x, -y, z],
            [x, y, z],
            [-x, y, z],
        ],
        dtype=float,
    )
    rotation = yaw_to_rotation_matrix(yaw)
    return local.dot(rotation.T) + center_arr


def project_3d_box_to_image(
    center: np.ndarray,
    dimensions: np.ndarray,
    yaw: float,
    calibration: CameraCalibration,
) -> Optional[np.ndarray]:
    """Project 3D box corners to image coordinates for debugging."""
    if calibration is None:
        return None
    corners = create_3d_box_corners(center, dimensions, yaw)
    projected = []
    for corner in corners:
        point = project_world_to_image(
            corner,
            camera_matrix=calibration.camera_matrix,
            intrinsic_matrix=calibration.intrinsic_matrix,
            extrinsic_matrix=calibration.extrinsic_matrix,
        )
        if point is None:
            return None
        projected.append(point)
    return np.asarray(projected, dtype=float)

