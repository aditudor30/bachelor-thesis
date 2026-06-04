"""Yaw estimation helpers for pseudo-3D."""

import math
from typing import Any, Dict, List, Optional

from deep_oc_sort_3d.pseudo3d.pseudo3d_types import Pseudo3DOutput


def estimate_default_yaw(config: Dict[str, Any]) -> float:
    """Return configured class/default yaw."""
    return float(config.get("class_default_yaw", 0.0))


def estimate_yaw_from_motion(outputs: List[Pseudo3DOutput], config: Dict[str, Any]) -> Optional[float]:
    """Estimate yaw from first and last valid centers in a track."""
    min_disp = float(config.get("min_track_displacement_for_yaw", 0.5))
    valid = [output for output in outputs if output.center_3d is not None]
    if len(valid) < 2:
        return None
    first = valid[0].center_3d
    last = valid[-1].center_3d
    if first is None or last is None:
        return None
    dx = float(last[0] - first[0])
    dy = float(last[1] - first[1])
    distance = math.sqrt(dx * dx + dy * dy)
    if distance < min_disp:
        return None
    return math.atan2(dy, dx)
