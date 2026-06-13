"""Circular yaw smoothing with conservative vehicle heading support."""

import math
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from deep_oc_sort_3d.official_023_027.official_track1_io import OfficialTrack1Row
from deep_oc_sort_3d.v4_geometry_refinement.track1_geometry_io import clone_row, group_tracks, position, progress_iter


def refine_track_yaw(
    rows: Sequence[OfficialTrack1Row],
    config: Dict[str, Any],
    progress: bool = True,
) -> Tuple[List[OfficialTrack1Row], List[Dict[str, Any]]]:
    """Smooth yaw circularly and blend stable heading for vehicle-like classes."""
    rules = config.get("yaw_refinement", {})
    if not bool(rules.get("enabled", True)):
        return list(rows), []
    heading_classes = set(int(value) for value in rules.get("heading_classes", [1, 2, 3, 6]))
    window = max(3, int(rules.get("yaw_smoothing_window", 5)))
    if window % 2 == 0:
        window += 1
    max_gap = int(rules.get("max_gap_for_smoothing", 10))
    max_delta = float(rules.get("max_yaw_change_per_frame_rad", 1.0))
    output = []
    changes = []
    for key, track in progress_iter(list(group_tracks(rows).items()), progress, "V4 yaw refinement"):
        yaws = np.asarray([normalize_angle(float(row.yaw)) for row in track], dtype=float)
        frames = [int(row.frame_id) for row in track]
        points = np.asarray([position(row) for row in track], dtype=float)
        refined = yaws.copy()
        for index in range(len(track)):
            indices = _yaw_window(frames, index, window, max_gap)
            if len(indices) >= 3:
                refined[index] = circular_median(yaws[indices])
        total_displacement = float(np.linalg.norm(points[-1, :2] - points[0, :2])) if len(track) > 1 else 0.0
        use_heading = int(key[1]) in heading_classes and bool(rules.get("use_heading_for_vehicle_like_classes", True)) and total_displacement >= float(rules.get("min_total_displacement_for_heading_m", 2.0))
        if int(key[1]) == 0 and not bool(rules.get("use_heading_for_person", False)):
            use_heading = False
        if use_heading:
            blend = float(rules.get("heading_blend", 0.25))
            for index in range(len(track)):
                heading = _local_heading(points, index, float(rules.get("min_step_for_heading_m", 0.25)))
                if heading is not None:
                    refined[index] = blend_angles(refined[index], heading, blend)
        for index in range(1, len(refined)):
            allowed = max_delta * float(max(1, frames[index] - frames[index - 1]))
            delta = angle_delta(refined[index - 1], refined[index])
            if abs(delta) > allowed:
                refined[index] = normalize_angle(refined[index - 1] + math.copysign(allowed, delta))
        for index, row in enumerate(track):
            value = normalize_angle(float(refined[index]))
            output.append(clone_row(row, yaw=value))
            change = abs(angle_delta(float(row.yaw), value))
            if change > 1e-9:
                changes.append({
                    "scene_id": key[0], "class_id": key[1], "object_id": key[2], "frame_id": row.frame_id,
                    "field_group": "yaw", "reason": "circular_smoothing_heading_blend" if use_heading else "circular_smoothing",
                    "old_yaw": float(row.yaw), "new_yaw": value, "yaw_change_rad": change,
                })
    return sorted(output, key=lambda row: row.key()), changes


def normalize_angle(value: float) -> float:
    """Normalize an angle to [-pi, pi]."""
    return float((value + math.pi) % (2.0 * math.pi) - math.pi)


def angle_delta(a: float, b: float) -> float:
    """Return signed shortest angular difference b-a."""
    return normalize_angle(float(b) - float(a))


def circular_median(values: Sequence[float]) -> float:
    """Return the observed angle minimizing total circular distance."""
    candidates = [normalize_angle(float(value)) for value in values]
    if not candidates:
        return 0.0
    return min(candidates, key=lambda candidate: sum(abs(angle_delta(candidate, value)) for value in candidates))


def blend_angles(a: float, b: float, weight_b: float) -> float:
    """Blend angles along their shortest circular arc."""
    return normalize_angle(float(a) + max(0.0, min(1.0, float(weight_b))) * angle_delta(a, b))


def _yaw_window(frames: Sequence[int], center: int, window: int, max_gap: int) -> List[int]:
    half = window // 2
    left = center
    right = center
    while left > max(0, center - half) and frames[left] - frames[left - 1] <= max_gap:
        left -= 1
    while right < min(len(frames) - 1, center + half) and frames[right + 1] - frames[right] <= max_gap:
        right += 1
    return list(range(left, right + 1))


def _local_heading(points: np.ndarray, index: int, minimum_step: float) -> Any:
    if len(points) < 2:
        return None
    left = max(0, index - 1)
    right = min(len(points) - 1, index + 1)
    vector = points[right, :2] - points[left, :2]
    if float(np.linalg.norm(vector)) < minimum_step:
        return None
    return math.atan2(float(vector[1]), float(vector[0]))
