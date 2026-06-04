"""Smoothness audit for Track 1 object trajectories."""

import math
from typing import Any, Dict, List, Optional, Tuple

from deep_oc_sort_3d.audit3d.audit3d_io import (
    finite_dimensions,
    finite_float,
    finite_xyz,
    group_rows,
    numeric_stats,
    percentile,
    progress_iter,
)


def group_track1_rows_by_object(rows: List[Dict[str, Any]]) -> Dict[Tuple[Any, Any, Any], List[Dict[str, Any]]]:
    """Group Track 1 rows by scene, class, and object id."""
    return group_rows(rows, ["scene_id", "class_id", "object_id"])


def compute_track_smoothness(rows_for_object: List[Dict[str, Any]], jump_threshold: float = 3.0) -> Dict[str, Any]:
    """Compute temporal smoothness metrics for one Track 1 object."""
    rows = sorted(rows_for_object, key=lambda row: int(row.get("frame_id") or 0))
    if not rows:
        return _empty_smoothness()

    frames = [int(row.get("frame_id") or 0) for row in rows]
    frame_gaps = [frames[index] - frames[index - 1] for index in range(1, len(frames))]
    step_distances = []
    jumps = []
    for index in range(1, len(rows)):
        previous = finite_xyz(rows[index - 1], "x", "y", "z")
        current = finite_xyz(rows[index], "x", "y", "z")
        if previous is None or current is None:
            continue
        distance = _distance(previous, current)
        step_distances.append(distance)
        jumps.append((rows[index - 1], rows[index], distance))

    first_xyz = _first_valid_xyz(rows)
    last_xyz = _last_valid_xyz(rows)
    travel_distance = sum(step_distances)
    straight_line_distance = _distance(first_xyz, last_xyz) if first_xyz is not None and last_xyz is not None else None
    path_efficiency = None
    if straight_line_distance is not None:
        if travel_distance > 0.0:
            path_efficiency = straight_line_distance / travel_distance
        elif straight_line_distance == 0.0:
            path_efficiency = 1.0

    yaw_deltas = _yaw_deltas(rows)
    dim_width = _dimension_values(rows, "width")
    dim_length = _dimension_values(rows, "length")
    dim_height = _dimension_values(rows, "height")

    return {
        "num_frames": len(rows),
        "frame_start": min(frames),
        "frame_end": max(frames),
        "mean_frame_gap": _mean(frame_gaps),
        "max_frame_gap": max(frame_gaps) if frame_gaps else 0,
        "step_count": len(step_distances),
        "step_distance_mean": _mean(step_distances),
        "step_distance_median": percentile(sorted(step_distances), 0.50) if step_distances else None,
        "step_distance_p95": percentile(sorted(step_distances), 0.95) if step_distances else None,
        "step_distance_max": max(step_distances) if step_distances else None,
        "travel_distance": travel_distance if step_distances else None,
        "straight_line_distance": straight_line_distance,
        "path_efficiency": path_efficiency,
        "jump_count": sum(1 for distance in step_distances if distance > jump_threshold),
        "jump_ratio": (
            float(sum(1 for distance in step_distances if distance > jump_threshold)) / float(len(step_distances))
            if step_distances
            else 0.0
        ),
        "yaw_delta_mean": _mean(yaw_deltas),
        "yaw_delta_max": max(yaw_deltas) if yaw_deltas else None,
        "dimension_std_width": _std(dim_width),
        "dimension_std_length": _std(dim_length),
        "dimension_std_height": _std(dim_height),
        "dimension_cv_width": _cv(dim_width),
        "dimension_cv_length": _cv(dim_length),
        "dimension_cv_height": _cv(dim_height),
        "missing_center_count": sum(1 for row in rows if finite_xyz(row, "x", "y", "z") is None),
        "missing_dimension_count": sum(1 for row in rows if finite_dimensions(row, "width", "length", "height") is None),
    }


def compute_smoothness_audit(
    rows: List[Dict[str, Any]],
    config: Dict[str, Any],
    show_progress: bool = True,
) -> Dict[str, Any]:
    """Compute smoothness metrics for all Track 1 objects."""
    grouped = group_track1_rows_by_object(rows)
    suspicious_step = float(config.get("suspicious_step_m", 3.0))
    invalid_step = float(config.get("invalid_step_m", 6.0))
    suspicious_cv = float(config.get("suspicious_dimension_cv", 0.25))
    invalid_cv = float(config.get("invalid_dimension_cv", 0.50))
    yaw_jump_threshold = float(config.get("yaw_jump_threshold", 1.57))
    per_object = []
    for key, group in progress_iter(list(grouped.items()), show_progress, "audit 3D smoothness", "track"):
        summary = compute_track_smoothness(group)
        summary["scene_id"] = key[0]
        summary["class_id"] = key[1]
        summary["object_id"] = key[2]
        max_step = summary.get("step_distance_max")
        max_cv = max(
            _none_to_zero(summary.get("dimension_cv_width")),
            _none_to_zero(summary.get("dimension_cv_length")),
            _none_to_zero(summary.get("dimension_cv_height")),
        )
        yaw_delta_max = _none_to_zero(summary.get("yaw_delta_max"))
        jump_count = _count_jumps(group, suspicious_step)
        invalid_jump_count = _count_jumps(group, invalid_step)
        summary["jump_count"] = jump_count
        summary["invalid_jump_count"] = invalid_jump_count
        summary["jump_ratio"] = float(jump_count) / float(summary.get("step_count") or 1)
        summary["status"] = _smoothness_status(max_step, max_cv, yaw_delta_max, invalid_step, suspicious_step, invalid_cv, suspicious_cv, yaw_jump_threshold)
        per_object.append(summary)

    return {
        "object_count": len(per_object),
        "row_count": len(rows),
        "thresholds": {
            "suspicious_step_m": suspicious_step,
            "invalid_step_m": invalid_step,
            "suspicious_dimension_cv": suspicious_cv,
            "invalid_dimension_cv": invalid_cv,
            "yaw_jump_threshold": yaw_jump_threshold,
        },
        "status_distribution": _distribution(per_object, "status"),
        "step_distance_max_stats": numeric_stats([row.get("step_distance_max") for row in per_object]),
        "jump_count_stats": numeric_stats([row.get("jump_count") for row in per_object]),
        "dimension_cv_max_stats": numeric_stats([_max_dimension_cv(row) for row in per_object]),
        "per_object": per_object,
    }


def find_worst_jumps(rows: List[Dict[str, Any]], top_k: int = 100) -> List[Dict[str, Any]]:
    """Return the largest consecutive 3D jumps."""
    jumps = []
    grouped = group_track1_rows_by_object(rows)
    for key, group in grouped.items():
        sorted_group = sorted(group, key=lambda row: int(row.get("frame_id") or 0))
        for index in range(1, len(sorted_group)):
            previous = finite_xyz(sorted_group[index - 1], "x", "y", "z")
            current = finite_xyz(sorted_group[index], "x", "y", "z")
            if previous is None or current is None:
                continue
            distance = _distance(previous, current)
            jumps.append(
                {
                    "scene_id": key[0],
                    "class_id": key[1],
                    "object_id": key[2],
                    "frame_id_prev": sorted_group[index - 1].get("frame_id"),
                    "frame_id": sorted_group[index].get("frame_id"),
                    "step_distance_3d": distance,
                    "x_prev": previous[0],
                    "y_prev": previous[1],
                    "z_prev": previous[2],
                    "x": current[0],
                    "y": current[1],
                    "z": current[2],
                }
            )
    return sorted(jumps, key=lambda row: float(row.get("step_distance_3d") or 0.0), reverse=True)[:top_k]


def find_worst_dimension_variation(rows: List[Dict[str, Any]], top_k: int = 100) -> List[Dict[str, Any]]:
    """Return tracks with the largest dimension coefficient of variation."""
    output = []
    grouped = group_track1_rows_by_object(rows)
    for key, group in grouped.items():
        summary = compute_track_smoothness(group)
        output.append(
            {
                "scene_id": key[0],
                "class_id": key[1],
                "object_id": key[2],
                "num_frames": summary.get("num_frames"),
                "dimension_cv_width": summary.get("dimension_cv_width"),
                "dimension_cv_length": summary.get("dimension_cv_length"),
                "dimension_cv_height": summary.get("dimension_cv_height"),
                "dimension_cv_max": _max_dimension_cv(summary),
                "dimension_std_width": summary.get("dimension_std_width"),
                "dimension_std_length": summary.get("dimension_std_length"),
                "dimension_std_height": summary.get("dimension_std_height"),
            }
        )
    return sorted(output, key=lambda row: float(row.get("dimension_cv_max") or 0.0), reverse=True)[:top_k]


def _empty_smoothness() -> Dict[str, Any]:
    return {
        "num_frames": 0,
        "frame_start": None,
        "frame_end": None,
        "mean_frame_gap": None,
        "max_frame_gap": None,
    }


def _distance(a: Tuple[float, float, float], b: Tuple[float, float, float]) -> float:
    return math.sqrt((a[0] - b[0]) ** 2.0 + (a[1] - b[1]) ** 2.0 + (a[2] - b[2]) ** 2.0)


def _mean(values: List[float]) -> Optional[float]:
    if not values:
        return None
    return sum(values) / float(len(values))


def _std(values: List[float]) -> Optional[float]:
    if not values:
        return None
    mean = _mean(values)
    if mean is None:
        return None
    return math.sqrt(sum((value - mean) ** 2.0 for value in values) / float(len(values)))


def _cv(values: List[float]) -> Optional[float]:
    mean = _mean(values)
    std = _std(values)
    if mean is None or std is None or abs(mean) < 1e-12:
        return None
    return std / abs(mean)


def _dimension_values(rows: List[Dict[str, Any]], field: str) -> List[float]:
    values = []
    for row in rows:
        value = finite_float(row.get(field))
        if value is not None:
            values.append(value)
    return values


def _yaw_deltas(rows: List[Dict[str, Any]]) -> List[float]:
    deltas = []
    sorted_rows = sorted(rows, key=lambda row: int(row.get("frame_id") or 0))
    previous = None
    for row in sorted_rows:
        yaw = finite_float(row.get("yaw"))
        if yaw is None:
            continue
        if previous is not None:
            deltas.append(abs(_angle_delta(previous, yaw)))
        previous = yaw
    return deltas


def _angle_delta(a: float, b: float) -> float:
    delta = b - a
    while delta > math.pi:
        delta -= 2.0 * math.pi
    while delta < -math.pi:
        delta += 2.0 * math.pi
    return delta


def _first_valid_xyz(rows: List[Dict[str, Any]]) -> Optional[Tuple[float, float, float]]:
    for row in rows:
        xyz = finite_xyz(row, "x", "y", "z")
        if xyz is not None:
            return xyz
    return None


def _last_valid_xyz(rows: List[Dict[str, Any]]) -> Optional[Tuple[float, float, float]]:
    for row in reversed(rows):
        xyz = finite_xyz(row, "x", "y", "z")
        if xyz is not None:
            return xyz
    return None


def _count_jumps(rows: List[Dict[str, Any]], threshold: float) -> int:
    count = 0
    sorted_rows = sorted(rows, key=lambda row: int(row.get("frame_id") or 0))
    for index in range(1, len(sorted_rows)):
        previous = finite_xyz(sorted_rows[index - 1], "x", "y", "z")
        current = finite_xyz(sorted_rows[index], "x", "y", "z")
        if previous is None or current is None:
            continue
        if _distance(previous, current) > threshold:
            count += 1
    return count


def _smoothness_status(
    max_step: Any,
    max_cv: float,
    yaw_delta_max: float,
    invalid_step: float,
    suspicious_step: float,
    invalid_cv: float,
    suspicious_cv: float,
    yaw_jump_threshold: float,
) -> str:
    max_step_value = _none_to_zero(max_step)
    if max_step_value > invalid_step or max_cv > invalid_cv:
        return "invalid"
    if max_step_value > suspicious_step or max_cv > suspicious_cv or yaw_delta_max > yaw_jump_threshold:
        return "suspicious"
    return "good"


def _distribution(rows: List[Dict[str, Any]], key: str) -> Dict[str, int]:
    counts = {}
    for row in rows:
        value = str(row.get(key, ""))
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items(), key=lambda item: item[0]))


def _none_to_zero(value: Any) -> float:
    try:
        if value is None:
            return 0.0
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _max_dimension_cv(row: Dict[str, Any]) -> float:
    return max(
        _none_to_zero(row.get("dimension_cv_width")),
        _none_to_zero(row.get("dimension_cv_length")),
        _none_to_zero(row.get("dimension_cv_height")),
    )
