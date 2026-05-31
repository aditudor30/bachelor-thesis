"""Camera geometry helpers for SmartSpaces calibration data.

Convention used here:
- ``cameraMatrix`` is treated as a 3x4 projection matrix from world homogeneous
  coordinates to image homogeneous coordinates.
- ``extrinsicMatrix`` is assumed to map world homogeneous coordinates to camera
  homogeneous coordinates. ``camera_to_world`` uses its inverse.

TODO: Validate these conventions visually on the real SmartSpaces calibration
files and adjust if the challenge data uses camera-to-world extrinsics instead.
"""

from typing import Optional, Tuple

import numpy as np


def ensure_matrix(mat, shape: Optional[Tuple[int, int]] = None) -> Optional[np.ndarray]:
    """Convert input to a float numpy matrix and optionally validate its shape."""
    if mat is None:
        return None
    try:
        matrix = np.asarray(mat, dtype=float)
    except (TypeError, ValueError):
        return None
    if matrix.ndim != 2:
        return None
    if shape is not None and matrix.shape != shape:
        return None
    return matrix


def pixel_to_camera_ray(u: float, v: float, intrinsic_matrix: np.ndarray) -> np.ndarray:
    """Return the camera ray direction [x, y, 1] for an image pixel."""
    intrinsic = ensure_matrix(intrinsic_matrix, (3, 3))
    if intrinsic is None:
        raise ValueError("intrinsic_matrix must be a 3x3 matrix")

    fx = intrinsic[0, 0]
    fy = intrinsic[1, 1]
    cx = intrinsic[0, 2]
    cy = intrinsic[1, 2]
    if abs(fx) < 1e-12 or abs(fy) < 1e-12:
        raise ValueError("intrinsic focal lengths must be non-zero")
    return np.asarray([(u - cx) / fx, (v - cy) / fy, 1.0], dtype=float)


def pixel_depth_to_camera_point(
    u: float,
    v: float,
    depth: float,
    intrinsic_matrix: np.ndarray,
) -> np.ndarray:
    """Backproject a pixel with image-plane depth into camera coordinates."""
    ray = pixel_to_camera_ray(u, v, intrinsic_matrix)
    return ray * float(depth)


def camera_to_world(
    point_camera: np.ndarray,
    extrinsic_matrix: Optional[np.ndarray],
) -> np.ndarray:
    """Transform a camera-space point to world coordinates.

    If the extrinsic matrix is missing or invalid, the input point is returned as
    a float array. Callers that require true world coordinates should check for a
    valid extrinsic before using this function.
    """
    point = np.asarray(point_camera, dtype=float).reshape(3)
    extrinsic = ensure_matrix(extrinsic_matrix)
    if extrinsic is None:
        return point.copy()
    if extrinsic.shape == (3, 4):
        extrinsic_4x4 = np.eye(4, dtype=float)
        extrinsic_4x4[:3, :] = extrinsic
        extrinsic = extrinsic_4x4
    if extrinsic.shape != (4, 4):
        return point.copy()

    point_h = np.asarray([point[0], point[1], point[2], 1.0], dtype=float)
    world_h = np.linalg.inv(extrinsic).dot(point_h)
    if abs(world_h[3]) > 1e-12:
        world_h = world_h / world_h[3]
    return world_h[:3]


def world_to_camera(
    point_world: np.ndarray,
    extrinsic_matrix: Optional[np.ndarray],
) -> np.ndarray:
    """Transform a world-space point to camera coordinates."""
    point = np.asarray(point_world, dtype=float).reshape(3)
    extrinsic = ensure_matrix(extrinsic_matrix)
    if extrinsic is None:
        return point.copy()
    if extrinsic.shape == (3, 4):
        point_h = np.asarray([point[0], point[1], point[2], 1.0], dtype=float)
        camera_h = extrinsic.dot(point_h)
        return camera_h[:3]
    if extrinsic.shape == (4, 4):
        point_h = np.asarray([point[0], point[1], point[2], 1.0], dtype=float)
        camera_h = extrinsic.dot(point_h)
        if abs(camera_h[3]) > 1e-12:
            camera_h = camera_h / camera_h[3]
        return camera_h[:3]
    return point.copy()


def project_world_to_image(
    point_world: np.ndarray,
    camera_matrix: Optional[np.ndarray] = None,
    intrinsic_matrix: Optional[np.ndarray] = None,
    extrinsic_matrix: Optional[np.ndarray] = None,
) -> Optional[Tuple[float, float]]:
    """Project a world point to image coordinates when calibration is sufficient."""
    point = np.asarray(point_world, dtype=float).reshape(3)

    projection = ensure_matrix(camera_matrix)
    if projection is not None:
        if projection.shape == (3, 4):
            point_h = np.asarray([point[0], point[1], point[2], 1.0], dtype=float)
            image_h = projection.dot(point_h)
        elif projection.shape == (3, 3):
            image_h = projection.dot(point)
        else:
            image_h = None
        if image_h is not None and abs(image_h[2]) > 1e-12:
            return (float(image_h[0] / image_h[2]), float(image_h[1] / image_h[2]))

    intrinsic = ensure_matrix(intrinsic_matrix, (3, 3))
    if intrinsic is None:
        return None

    if extrinsic_matrix is not None:
        point_camera = world_to_camera(point, extrinsic_matrix)
    else:
        # TODO: Use this fallback only for dummy tests or already-camera-space points.
        point_camera = point

    if abs(point_camera[2]) < 1e-12:
        return None
    image_h = intrinsic.dot(point_camera)
    return (float(image_h[0] / image_h[2]), float(image_h[1] / image_h[2]))

