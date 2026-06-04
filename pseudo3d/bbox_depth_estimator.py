"""BBox-height depth estimator for pseudo-3D."""

from typing import Any, Dict, Optional, Tuple

from deep_oc_sort_3d.pseudo3d.camera_model_adapter import CameraModel


def bbox_height_px(bbox_xyxy: Tuple[float, float, float, float]) -> float:
    """Return bbox height in pixels."""
    return max(0.0, float(bbox_xyxy[3]) - float(bbox_xyxy[1]))


def bbox_projection_point(
    bbox_xyxy: Tuple[float, float, float, float],
    projection_point: str = "bottom_center",
) -> Tuple[float, float]:
    """Return configured 2D point for backprojection."""
    x1, y1, x2, y2 = bbox_xyxy
    u = (float(x1) + float(x2)) * 0.5
    if projection_point == "center":
        return u, (float(y1) + float(y2)) * 0.5
    return u, float(y2)


def estimate_depth_from_bbox_height(
    bbox_xyxy: Tuple[float, float, float, float],
    real_height: float,
    camera_model: CameraModel,
    config: Dict[str, Any],
) -> Tuple[Optional[float], Optional[str]]:
    """Estimate depth from bbox height and class height prior."""
    height_px = bbox_height_px(bbox_xyxy)
    min_height = float(config.get("min_bbox_height_px", 8))
    if height_px < min_height:
        return None, "bbox_height_too_small"
    if real_height <= 0.0:
        return None, "invalid_height_prior"
    depth = camera_model.fy * float(real_height) / height_px
    min_depth = float(config.get("min_depth_m", 0.1))
    max_depth = float(config.get("max_depth_m", 100.0))
    depth = max(min_depth, min(max_depth, depth))
    return depth, None

