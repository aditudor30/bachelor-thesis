"""Depth-based projection and backprojection helpers."""

from typing import Optional, Tuple

import numpy as np

from deep_oc_sort_3d.data.calibration import CameraCalibration
from deep_oc_sort_3d.geometry.camera_geometry import (
    camera_to_world,
    ensure_matrix,
    pixel_depth_to_camera_point,
)


def sample_depth_at_pixel(
    depth: np.ndarray,
    u: float,
    v: float,
    window: int = 3,
) -> Optional[float]:
    """Sample robust median depth around a pixel."""
    if depth is None or depth.ndim < 2:
        return None
    height, width = depth.shape[:2]
    x = int(round(u))
    y = int(round(v))
    if x < 0 or y < 0 or x >= width or y >= height:
        return None

    radius = max(int(window), 1) // 2
    x0 = max(0, x - radius)
    x1 = min(width, x + radius + 1)
    y0 = max(0, y - radius)
    y1 = min(height, y + radius + 1)
    values = np.asarray(depth[y0:y1, x0:x1], dtype=float).reshape(-1)
    valid = values[np.isfinite(values) & (values > 0.0)]
    if valid.size == 0:
        return None
    return float(np.median(valid))


def median_depth_in_bbox(
    depth: np.ndarray,
    bbox_xyxy: Tuple[float, float, float, float],
    min_valid_depth: float = 1e-6,
) -> Optional[float]:
    """Return median valid depth inside an xyxy bounding box."""
    if depth is None or depth.ndim < 2:
        return None
    height, width = depth.shape[:2]
    x1, y1, x2, y2 = bbox_xyxy
    left = max(0, int(np.floor(min(x1, x2))))
    right = min(width, int(np.ceil(max(x1, x2))))
    top = max(0, int(np.floor(min(y1, y2))))
    bottom = min(height, int(np.ceil(max(y1, y2))))
    if right <= left or bottom <= top:
        return None

    values = np.asarray(depth[top:bottom, left:right], dtype=float).reshape(-1)
    valid = values[np.isfinite(values) & (values > min_valid_depth)]
    if valid.size == 0:
        return None
    return float(np.median(valid))


def bbox_center(bbox_xyxy: Tuple[float, float, float, float]) -> Tuple[float, float]:
    """Return the center of an xyxy bounding box."""
    x1, y1, x2, y2 = bbox_xyxy
    return ((float(x1) + float(x2)) * 0.5, (float(y1) + float(y2)) * 0.5)


def bbox_bottom_center(bbox_xyxy: Tuple[float, float, float, float]) -> Tuple[float, float]:
    """Return the bottom-center point of an xyxy bounding box."""
    x1, _y1, x2, y2 = bbox_xyxy
    return ((float(x1) + float(x2)) * 0.5, float(y2))


def backproject_bbox_to_world(
    bbox_xyxy: Tuple[float, float, float, float],
    depth: np.ndarray,
    calibration: CameraCalibration,
    use_bottom_center: bool = True,
    depth_stat: str = "median",
) -> Optional[np.ndarray]:
    """Backproject a visible object bbox to a rough world-space point."""
    if depth is None or calibration is None:
        return None
    intrinsic = ensure_matrix(calibration.intrinsic_matrix, (3, 3))
    extrinsic = ensure_matrix(calibration.extrinsic_matrix)
    if intrinsic is None or extrinsic is None:
        return None

    if use_bottom_center:
        u, v = bbox_bottom_center(bbox_xyxy)
    else:
        u, v = bbox_center(bbox_xyxy)

    if depth_stat == "point":
        depth_value = sample_depth_at_pixel(depth, u, v)
    else:
        depth_value = median_depth_in_bbox(depth, bbox_xyxy)
    if depth_value is None:
        return None

    point_camera = pixel_depth_to_camera_point(u, v, depth_value, intrinsic)
    return camera_to_world(point_camera, extrinsic)

