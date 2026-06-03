"""Draw 2D boxes and projected 3D cuboids on RGB frames."""

from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from deep_oc_sort_3d.visualization3d.cuboid_geometry import get_cuboid_edges
from deep_oc_sort_3d.visualization3d.cuboid_projection import project_cuboid_to_image
from deep_oc_sort_3d.visualization3d.visualization_io import (
    parse_bbox_xyxy_from_record,
    parse_center_dimensions_yaw_from_record,
)


def draw_2d_bbox(image: np.ndarray, bbox_xyxy, label: str) -> np.ndarray:
    """Draw one 2D bounding box on an RGB image."""
    output = image.copy()
    color = (255, 220, 0)
    x1, y1, x2, y2 = bbox_xyxy
    p1 = (int(round(x1)), int(round(y1)))
    p2 = (int(round(x2)), int(round(y2)))
    cv2.rectangle(output, p1, p2, color, 2)
    _draw_label(output, p1, label, color)
    return output


def draw_projected_cuboid(image: np.ndarray, points_2d, label: Optional[str] = None) -> np.ndarray:
    """Draw a projected 3D cuboid from its 8 image-space corners."""
    output = image.copy()
    points = np.asarray(points_2d, dtype=float)
    if points.shape != (8, 2) or not np.all(np.isfinite(points)):
        return output
    color = (0, 255, 90)
    for start, end in get_cuboid_edges():
        p1 = (int(round(points[start, 0])), int(round(points[start, 1])))
        p2 = (int(round(points[end, 0])), int(round(points[end, 1])))
        cv2.line(output, p1, p2, color, 2)
    if label:
        anchor = (int(round(points[0, 0])), int(round(points[0, 1])))
        _draw_label(output, anchor, label, color)
    return output


def draw_global_frame_records(
    image: np.ndarray,
    records: List[Dict[str, Any]],
    calibration: Optional[Any] = None,
    draw_2d: bool = True,
    draw_3d: bool = True,
    draw_labels: bool = True,
) -> Tuple[np.ndarray, Dict[str, int]]:
    """Draw frame records and return the annotated image plus a summary."""
    output = image.copy()
    summary = {
        "records": 0,
        "bbox_drawn": 0,
        "cuboid_projected": 0,
        "cuboid_failed": 0,
    }
    for record in records:
        summary["records"] += 1
        label = _record_label(record) if draw_labels else ""
        bbox = parse_bbox_xyxy_from_record(record)
        if draw_2d and bbox is not None:
            output = draw_2d_bbox(output, bbox, label)
            summary["bbox_drawn"] += 1
        if draw_3d:
            parsed = parse_center_dimensions_yaw_from_record(record)
            if parsed is None or calibration is None:
                summary["cuboid_failed"] += 1
                continue
            center, dimensions, yaw = parsed
            projection = project_cuboid_to_image(center, dimensions, yaw, calibration)
            if projection.get("success"):
                output = draw_projected_cuboid(output, projection.get("points_2d"), label if not draw_2d else None)
                summary["cuboid_projected"] += 1
            else:
                summary["cuboid_failed"] += 1
    return output, summary


def _record_label(record: Dict[str, Any]) -> str:
    global_id = record.get("global_track_id", "")
    class_name = str(record.get("class_name", ""))
    confidence = _safe_float(record.get("confidence"))
    if confidence is None:
        return "G%s %s" % (global_id, class_name)
    return "G%s %s %.2f" % (global_id, class_name, confidence)


def _draw_label(image: np.ndarray, anchor: Tuple[int, int], label: str, color: Tuple[int, int, int]) -> None:
    if not label:
        return
    x, y = anchor
    y = max(14, y)
    cv2.putText(image, label, (x, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


def _safe_float(value: Any) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None

