"""Robust adapter around SmartSpaces calibration for pseudo-3D estimation."""

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np

from deep_oc_sort_3d.geometry.camera_geometry import camera_to_world, ensure_matrix, pixel_depth_to_camera_point


@dataclass
class CameraModel:
    """Minimal camera model needed by pseudo-3D estimation."""

    intrinsic_matrix: np.ndarray
    extrinsic_matrix: Optional[np.ndarray]
    camera_matrix: Optional[np.ndarray]
    frame_width: Optional[int]
    frame_height: Optional[int]

    @property
    def fx(self) -> float:
        return float(self.intrinsic_matrix[0, 0])

    @property
    def fy(self) -> float:
        return float(self.intrinsic_matrix[1, 1])

    @property
    def cx(self) -> float:
        return float(self.intrinsic_matrix[0, 2])

    @property
    def cy(self) -> float:
        return float(self.intrinsic_matrix[1, 2])


def camera_model_from_calibration(calibration: Any) -> Tuple[Optional[CameraModel], Optional[str]]:
    """Extract a minimal camera model from dict or calibration dataclass."""
    intrinsic = _matrix(calibration, "intrinsic_matrix", "intrinsicMatrix")
    if intrinsic is None:
        camera_matrix = _matrix(calibration, "camera_matrix", "cameraMatrix")
        if camera_matrix is not None and camera_matrix.shape in ((3, 3), (3, 4)):
            intrinsic = camera_matrix[:, :3]
    if intrinsic is None or intrinsic.shape != (3, 3):
        return None, "missing_intrinsics"
    if abs(float(intrinsic[0, 0])) < 1e-12 or abs(float(intrinsic[1, 1])) < 1e-12:
        return None, "invalid_intrinsics"
    return (
        CameraModel(
            intrinsic_matrix=intrinsic,
            extrinsic_matrix=_matrix(calibration, "extrinsic_matrix", "extrinsicMatrix"),
            camera_matrix=_matrix(calibration, "camera_matrix", "cameraMatrix"),
            frame_width=_optional_int(_value(calibration, "frame_width", "frameWidth")),
            frame_height=_optional_int(_value(calibration, "frame_height", "frameHeight")),
        ),
        None,
    )


def backproject_pixel_with_depth(
    u: float,
    v: float,
    depth: float,
    camera_model: CameraModel,
    require_world_coordinates: bool = False,
) -> Tuple[Optional[np.ndarray], str, Optional[str]]:
    """Backproject a pixel with estimated depth to world or camera coordinates."""
    try:
        point_camera = pixel_depth_to_camera_point(u, v, depth, camera_model.intrinsic_matrix)
    except Exception:
        return None, "unknown", "backprojection_failed"
    if camera_model.extrinsic_matrix is None:
        if require_world_coordinates:
            return None, "camera", "missing_extrinsics"
        return point_camera, "camera", None
    try:
        return camera_to_world(point_camera, camera_model.extrinsic_matrix), "world", None
    except Exception:
        if require_world_coordinates:
            return None, "camera", "world_transform_failed"
        return point_camera, "camera", "world_transform_failed"


def _matrix(calibration: Any, snake_key: str, json_key: str) -> Optional[np.ndarray]:
    value = _value(calibration, snake_key, json_key)
    matrix = ensure_matrix(value)
    return matrix


def _value(calibration: Any, snake_key: str, json_key: str) -> Any:
    if calibration is None:
        return None
    if isinstance(calibration, dict):
        value = calibration.get(snake_key)
        if value is None:
            value = calibration.get(json_key)
        return value
    value = getattr(calibration, snake_key, None)
    if value is None:
        value = getattr(calibration, json_key, None)
    return value


def _optional_int(value: Any) -> Optional[int]:
    if value in (None, ""):
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None

