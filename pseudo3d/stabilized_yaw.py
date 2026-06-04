"""Yaw estimation from stabilized pseudo-3D motion."""

import math
from typing import List, Optional, Tuple

import numpy as np


def estimate_yaw_from_smoothed_motion(
    centers: np.ndarray,
    frame_ids: List[int],
    min_displacement: float = 0.5,
    default_yaw: float = 0.0,
) -> Tuple[List[float], List[str]]:
    """Estimate yaw from local motion direction after center smoothing."""
    array = np.asarray(centers, dtype=float).reshape(-1, 3)
    yaws = []
    sources = []
    last_valid_index = None
    last_motion_yaw = None
    for index in range(array.shape[0]):
        current = array[index]
        yaw = float(default_yaw)
        source = "class_default"
        if np.all(np.isfinite(current)):
            motion = _motion_from_neighbors(array, index, last_valid_index)
            if motion is not None:
                dx, dy = motion
                displacement = math.sqrt(dx * dx + dy * dy)
                if displacement >= float(min_displacement):
                    yaw = math.atan2(dy, dx)
                    source = "motion_direction_smoothed"
                    last_motion_yaw = yaw
            elif last_motion_yaw is not None:
                yaw = last_motion_yaw
                source = "motion_direction_smoothed"
            last_valid_index = index
        yaws.append(float(yaw))
        sources.append(source)
    return yaws, sources


def _motion_from_neighbors(array: np.ndarray, index: int, last_valid_index: Optional[int]) -> Optional[Tuple[float, float]]:
    current = array[index]
    if last_valid_index is not None and np.all(np.isfinite(array[last_valid_index])):
        previous = array[last_valid_index]
        return float(current[0] - previous[0]), float(current[1] - previous[1])
    for next_index in range(index + 1, array.shape[0]):
        following = array[next_index]
        if np.all(np.isfinite(following)):
            return float(following[0] - current[0]), float(following[1] - current[1])
    return None
