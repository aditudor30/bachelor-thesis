"""Jump detection and correction for pseudo-3D center sequences."""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def compute_step_distances(centers: np.ndarray) -> np.ndarray:
    """Return center-to-center step distances, with NaN for invalid steps."""
    array = _center_array(centers)
    distances = np.full((array.shape[0],), np.nan, dtype=float)
    for index in range(1, array.shape[0]):
        prev = array[index - 1]
        curr = array[index]
        if np.all(np.isfinite(prev)) and np.all(np.isfinite(curr)):
            distances[index] = float(np.linalg.norm(curr - prev))
    return distances


def detect_jump_outliers(
    centers: np.ndarray,
    frame_ids: List[int],
    max_step_m: float,
    max_speed_m_per_frame: Optional[float] = None,
) -> Dict[str, Any]:
    """Detect jumps above a distance or per-frame speed threshold."""
    array = _center_array(centers)
    distances = compute_step_distances(array)
    jump_indices = []
    for index in range(1, array.shape[0]):
        distance = distances[index]
        if not np.isfinite(distance):
            continue
        frame_delta = _frame_delta(frame_ids, index)
        limit = float(max_step_m)
        if max_speed_m_per_frame is not None:
            limit = min(limit, float(max_speed_m_per_frame) * float(frame_delta))
        if distance > limit:
            jump_indices.append(index)
    finite = distances[np.isfinite(distances)]
    return {
        "num_jumps": len(jump_indices),
        "jump_indices": jump_indices,
        "max_step_before": float(np.max(finite)) if finite.size else None,
        "step_distances": distances.tolist(),
    }


def apply_jump_guard(
    centers: np.ndarray,
    frame_ids: List[int],
    max_step_m: float,
    strategy: str = "hold_previous",
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Correct large jumps with hold, interpolation, or invalidation."""
    corrected = _center_array(centers).copy()
    before = detect_jump_outliers(corrected, frame_ids, max_step_m)
    corrected_indices = []
    for index in before["jump_indices"]:
        if strategy == "mark_invalid":
            corrected[index, :] = np.nan
        elif strategy == "interpolate":
            corrected[index, :] = _interpolate_point(corrected, index)
        else:
            corrected[index, :] = _previous_valid(corrected, index)
        corrected_indices.append(index)
    after_distances = compute_step_distances(corrected)
    finite_after = after_distances[np.isfinite(after_distances)]
    report = dict(before)
    report.update(
        {
            "strategy": strategy,
            "corrected_indices": corrected_indices,
            "max_step_after": float(np.max(finite_after)) if finite_after.size else None,
            "step_distances_after": after_distances.tolist(),
        }
    )
    return corrected, report


def _center_array(centers: np.ndarray) -> np.ndarray:
    array = np.asarray(centers, dtype=float)
    if array.size == 0:
        return array.reshape(0, 3)
    return array.reshape(-1, 3)


def _frame_delta(frame_ids: List[int], index: int) -> int:
    if index >= len(frame_ids):
        return 1
    try:
        delta = int(frame_ids[index]) - int(frame_ids[index - 1])
    except (TypeError, ValueError):
        return 1
    return max(1, delta)


def _previous_valid(centers: np.ndarray, index: int) -> np.ndarray:
    for prev_index in range(index - 1, -1, -1):
        if np.all(np.isfinite(centers[prev_index])):
            return centers[prev_index].copy()
    return np.full((3,), np.nan, dtype=float)


def _next_valid(centers: np.ndarray, index: int) -> np.ndarray:
    for next_index in range(index + 1, centers.shape[0]):
        if np.all(np.isfinite(centers[next_index])):
            return centers[next_index].copy()
    return np.full((3,), np.nan, dtype=float)


def _interpolate_point(centers: np.ndarray, index: int) -> np.ndarray:
    previous = _previous_valid(centers, index)
    following = _next_valid(centers, index)
    if np.all(np.isfinite(previous)) and np.all(np.isfinite(following)):
        return (previous + following) / 2.0
    if np.all(np.isfinite(previous)):
        return previous
    return following
