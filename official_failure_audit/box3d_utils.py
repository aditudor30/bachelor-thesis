"""Small 3D box helpers used by matching and hypothesis transforms."""

import math
from typing import Optional

import numpy as np

from deep_oc_sort_3d.official_failure_audit.track1_parser import AuditTrack1Row


def center(row: AuditTrack1Row) -> np.ndarray:
    return np.asarray([row.x, row.y, row.z], dtype=float)


def dimensions(row: AuditTrack1Row) -> np.ndarray:
    return np.asarray([row.width, row.length, row.height], dtype=float)


def center_distance(a: AuditTrack1Row, b: AuditTrack1Row) -> float:
    return float(np.linalg.norm(center(a) - center(b)))


def yaw_error(a: float, b: float) -> float:
    return abs(float((float(b) - float(a) + math.pi) % (2.0 * math.pi) - math.pi))


def normalize_yaw(value: float) -> float:
    return float((float(value) + math.pi) % (2.0 * math.pi) - math.pi)


def axis_aligned_iou3d(a: AuditTrack1Row, b: AuditTrack1Row) -> Optional[float]:
    dims_a = dimensions(a)
    dims_b = dimensions(b)
    if np.any(dims_a <= 0.0) or np.any(dims_b <= 0.0):
        return None
    min_a = center(a) - dims_a / 2.0
    max_a = center(a) + dims_a / 2.0
    min_b = center(b) - dims_b / 2.0
    max_b = center(b) + dims_b / 2.0
    intersection = np.maximum(0.0, np.minimum(max_a, max_b) - np.maximum(min_a, min_b))
    intersection_volume = float(np.prod(intersection))
    union = float(np.prod(dims_a) + np.prod(dims_b) - intersection_volume)
    return None if union <= 0.0 else intersection_volume / union
