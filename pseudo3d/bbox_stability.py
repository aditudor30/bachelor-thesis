"""BBox stability helpers for pseudo-3D depth estimation."""

from typing import Any, Dict, List

from deep_oc_sort_3d.audit3d.audit3d_io import numeric_stats


def compute_bbox_height(bbox_xyxy: Any) -> float:
    """Return bbox height in pixels from x1,y1,x2,y2-like data."""
    if isinstance(bbox_xyxy, str):
        bbox_xyxy = _parse_bbox_string(bbox_xyxy)
    try:
        values = list(bbox_xyxy)
        return max(0.0, float(values[3]) - float(values[1]))
    except (TypeError, ValueError, IndexError):
        return 0.0


def mark_small_bbox_unstable(records: List[Any], min_height_px: float) -> List[bool]:
    """Mark records whose bbox height is too small for stable depth."""
    marks = []
    for record in records:
        bbox = getattr(record, "bbox_xyxy", None)
        if bbox is None and isinstance(record, dict):
            bbox = record.get("bbox_xyxy")
            if bbox is None:
                bbox = (record.get("x1"), record.get("y1"), record.get("x2"), record.get("y2"))
        marks.append(compute_bbox_height(bbox) < float(min_height_px))
    return marks


def compute_bbox_height_stats(records: List[Any]) -> Dict[str, Any]:
    """Compute numeric stats for bbox heights."""
    heights = []
    for record in records:
        bbox = getattr(record, "bbox_xyxy", None)
        if bbox is None and isinstance(record, dict):
            bbox = record.get("bbox_xyxy")
            if bbox is None:
                bbox = (record.get("x1"), record.get("y1"), record.get("x2"), record.get("y2"))
        heights.append(compute_bbox_height(bbox))
    return numeric_stats(heights)


def _parse_bbox_string(value: str) -> List[float]:
    text = value.strip().strip("[]()")
    if not text:
        return []
    parts = [part.strip() for part in text.replace(";", ",").split(",")]
    return [float(part) for part in parts if part]
