"""I/O helpers for MVP visualization records."""

import csv
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np


def load_global_frame_records_csv(path: Union[str, Path]) -> List[Dict[str, Any]]:
    """Load frame-level or generic global records from CSV."""
    rows = []
    with Path(path).open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(dict(row))
    return rows


def filter_records_by_frame(records: List[Dict[str, Any]], frame_id: int) -> List[Dict[str, Any]]:
    """Filter records by integer frame id."""
    return [record for record in records if _optional_int(record.get("frame_id")) == int(frame_id)]


def filter_records_by_global_track_id(records: List[Dict[str, Any]], global_track_id: int) -> List[Dict[str, Any]]:
    """Filter records by global track id."""
    return [record for record in records if _optional_int(record.get("global_track_id")) == int(global_track_id)]


def filter_records_by_class(records: List[Dict[str, Any]], class_name: str) -> List[Dict[str, Any]]:
    """Filter records by class name."""
    wanted = str(class_name)
    return [record for record in records if str(record.get("class_name", "")) == wanted]


def parse_center_dimensions_yaw_from_record(record: Dict[str, Any]) -> Optional[Tuple[np.ndarray, np.ndarray, float]]:
    """Parse center, dimensions, and yaw from a generic/global CSV record."""
    center = _optional_float_array([record.get("center_x"), record.get("center_y"), record.get("center_z")])
    dimensions = _optional_float_array([record.get("width_3d"), record.get("length_3d"), record.get("height_3d")])
    yaw = _optional_float(record.get("yaw"))
    if center is None or dimensions is None or yaw is None:
        return None
    if center.shape[0] != 3 or dimensions.shape[0] != 3:
        return None
    if not np.all(np.isfinite(center)) or not np.all(np.isfinite(dimensions)) or not np.isfinite(float(yaw)):
        return None
    return center, dimensions, float(yaw)


def parse_bbox_xyxy_from_record(record: Dict[str, Any]) -> Optional[Tuple[float, float, float, float]]:
    """Parse an xyxy 2D bbox from a record."""
    values = [_optional_float(record.get(key)) for key in ("x1", "y1", "x2", "y2")]
    if any(value is None for value in values):
        return None
    x1, y1, x2, y2 = [float(value) for value in values]
    if not all(np.isfinite([x1, y1, x2, y2])):
        return None
    if x2 <= x1 or y2 <= y1:
        return None
    return (x1, y1, x2, y2)


def _optional_float_array(values: List[Any]) -> Optional[np.ndarray]:
    parsed = []
    for value in values:
        item = _optional_float(value)
        if item is None:
            return None
        parsed.append(float(item))
    return np.asarray(parsed, dtype=float)


def _optional_float(value: Any) -> Optional[float]:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _optional_int(value: Any) -> Optional[int]:
    if value in (None, ""):
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None

