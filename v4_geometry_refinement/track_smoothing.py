"""Gap-aware robust xyz smoothing for V4."""

from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from deep_oc_sort_3d.official_023_027.official_track1_io import OfficialTrack1Row
from deep_oc_sort_3d.v4_geometry_refinement.track1_geometry_io import clone_row, group_tracks, position, progress_iter


def smooth_track_positions(
    rows: Sequence[OfficialTrack1Row],
    config: Dict[str, Any],
    progress: bool = True,
) -> Tuple[List[OfficialTrack1Row], List[Dict[str, Any]]]:
    """Apply bounded moving-median smoothing to sufficiently long tracks."""
    rules = config.get("smoothing", {})
    if not bool(rules.get("enabled", True)):
        return list(rows), []
    grouped = group_tracks(rows)
    output = []
    changes = []
    minimum = int(rules.get("min_track_length", 5))
    window = max(3, int(rules.get("xyz_window", 5)))
    if window % 2 == 0:
        window += 1
    preserve_endpoints = bool(rules.get("preserve_endpoints", True))
    max_gap = int(rules.get("max_gap_for_smoothing", 10))
    max_change = float(rules.get("max_position_change_per_point_m", 5.0))
    for key, track in progress_iter(list(grouped.items()), progress, "V4 xyz smoothing"):
        if len(track) < minimum:
            output.extend(track)
            continue
        points = np.asarray([position(row) for row in track], dtype=float)
        frames = [int(row.frame_id) for row in track]
        refined = points.copy()
        for index in range(len(track)):
            if preserve_endpoints and index in (0, len(track) - 1):
                continue
            indices = _contiguous_window_indices(frames, index, window, max_gap)
            if len(indices) < 3:
                continue
            target = np.median(points[indices], axis=0)
            delta = target - points[index]
            distance = float(np.linalg.norm(delta))
            if distance > max_change and distance > 0.0:
                target = points[index] + delta * (max_change / distance)
            refined[index] = target
        for index, row in enumerate(track):
            new_row = clone_row(row, x=float(refined[index, 0]), y=float(refined[index, 1]), z=float(refined[index, 2]))
            output.append(new_row)
            distance = float(np.linalg.norm(refined[index] - points[index]))
            if distance > 1e-9:
                changes.append(_position_change(key, row, points[index], refined[index], distance, "moving_median"))
    return sorted(output, key=lambda row: row.key()), changes


def _contiguous_window_indices(frames: Sequence[int], center: int, window: int, max_gap: int) -> List[int]:
    half = window // 2
    left = center
    right = center
    while left > max(0, center - half) and frames[left] - frames[left - 1] <= max_gap:
        left -= 1
    while right < min(len(frames) - 1, center + half) and frames[right + 1] - frames[right] <= max_gap:
        right += 1
    return list(range(left, right + 1))


def _position_change(key: Any, row: OfficialTrack1Row, old: np.ndarray, new: np.ndarray, distance: float, reason: str) -> Dict[str, Any]:
    return {
        "scene_id": key[0], "class_id": key[1], "object_id": key[2], "frame_id": row.frame_id,
        "field_group": "position", "reason": reason, "old_x": float(old[0]), "old_y": float(old[1]),
        "old_z": float(old[2]), "new_x": float(new[0]), "new_y": float(new[1]), "new_z": float(new[2]),
        "position_change_m": distance,
    }

