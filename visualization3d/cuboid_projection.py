"""Projection helpers for visualizing 3D cuboids on camera frames."""

from typing import Any, Dict, Optional

import numpy as np

from deep_oc_sort_3d.geometry.camera_geometry import ensure_matrix, project_world_to_image, world_to_camera
from deep_oc_sort_3d.visualization3d.cuboid_geometry import compute_3d_cuboid_corners, validate_cuboid_inputs


def project_points_to_image(points_3d: np.ndarray, calibration: Dict[str, Any]) -> np.ndarray:
    """Project 3D points to image coordinates.

    Missing or invalid calibration produces NaN rows instead of raising. A
    point with non-positive projected depth is treated as not projectable.
    """
    points = np.asarray(points_3d, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points_3d must have shape (N, 3)")

    camera_matrix = _calibration_matrix(calibration, "camera_matrix", "cameraMatrix")
    intrinsic_matrix = _calibration_matrix(calibration, "intrinsic_matrix", "intrinsicMatrix")
    extrinsic_matrix = _calibration_matrix(calibration, "extrinsic_matrix", "extrinsicMatrix")

    projected = []
    for point in points:
        image_point = _project_one_point(point, camera_matrix, intrinsic_matrix, extrinsic_matrix)
        if image_point is None:
            projected.append((float("nan"), float("nan")))
        else:
            projected.append(image_point)
    return np.asarray(projected, dtype=float)


def project_cuboid_to_image(
    center,
    dimensions,
    yaw: float,
    calibration: Dict[str, Any],
) -> Dict[str, Any]:
    """Project one 3D cuboid to image space.

    Returns a dictionary with ``success``, ``points_2d`` and ``error_message``.
    This is intended for debugging visualization, not as a training primitive.
    """
    if not validate_cuboid_inputs(center, dimensions, yaw):
        return {"success": False, "points_2d": None, "error_message": "invalid_cuboid_inputs"}
    try:
        corners = compute_3d_cuboid_corners(np.asarray(center, dtype=float), np.asarray(dimensions, dtype=float), float(yaw))
        points_2d = project_points_to_image(corners, calibration)
    except Exception as exc:
        return {"success": False, "points_2d": None, "error_message": str(exc)}
    if points_2d.shape != (8, 2) or not np.all(np.isfinite(points_2d)):
        return {"success": False, "points_2d": None, "error_message": "projection_failed"}
    return {"success": True, "points_2d": points_2d, "error_message": ""}


def is_projected_cuboid_visible(points_2d, image_width: int, image_height: int) -> bool:
    """Return True if any projected cuboid point falls inside the image frame."""
    if points_2d is None:
        return False
    points = np.asarray(points_2d, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2 or not np.any(np.isfinite(points)):
        return False
    xs = points[:, 0]
    ys = points[:, 1]
    inside_x = np.logical_and(xs >= 0.0, xs < float(image_width))
    inside_y = np.logical_and(ys >= 0.0, ys < float(image_height))
    return bool(np.any(np.logical_and(inside_x, inside_y)))


def _project_one_point(
    point: np.ndarray,
    camera_matrix: Optional[np.ndarray],
    intrinsic_matrix: Optional[np.ndarray],
    extrinsic_matrix: Optional[np.ndarray],
) -> Optional[Any]:
    if camera_matrix is not None:
        projected = _project_with_camera_matrix(point, camera_matrix)
        if projected is not None:
            return projected

    if intrinsic_matrix is None:
        return None
    if extrinsic_matrix is not None:
        point_camera = world_to_camera(point, extrinsic_matrix)
        if point_camera.shape[0] >= 3 and float(point_camera[2]) <= 1e-12:
            return None
    return project_world_to_image(
        point,
        camera_matrix=None,
        intrinsic_matrix=intrinsic_matrix,
        extrinsic_matrix=extrinsic_matrix,
    )


def _project_with_camera_matrix(point: np.ndarray, camera_matrix: np.ndarray) -> Optional[Any]:
    projection = ensure_matrix(camera_matrix)
    if projection is None:
        return None
    point_arr = np.asarray(point, dtype=float).reshape(3)
    if projection.shape == (3, 4):
        point_h = np.asarray([point_arr[0], point_arr[1], point_arr[2], 1.0], dtype=float)
        image_h = projection.dot(point_h)
    elif projection.shape == (3, 3):
        image_h = projection.dot(point_arr)
    else:
        return None
    if image_h.shape[0] < 3 or float(image_h[2]) <= 1e-12:
        return None
    return (float(image_h[0] / image_h[2]), float(image_h[1] / image_h[2]))


def _calibration_matrix(calibration: Any, snake_key: str, json_key: str) -> Optional[np.ndarray]:
    value = None
    if calibration is None:
        return None
    if isinstance(calibration, dict):
        value = calibration.get(snake_key)
        if value is None:
            value = calibration.get(json_key)
    else:
        value = getattr(calibration, snake_key, None)
    return ensure_matrix(value)

