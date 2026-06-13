"""Conservative isolated 3D outlier repair for V4."""

from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from deep_oc_sort_3d.official_023_027.official_track1_io import OfficialTrack1Row
from deep_oc_sort_3d.v4_geometry_refinement.track1_geometry_io import clone_row, group_tracks, position, progress_iter


def repair_position_outliers(
    rows: Sequence[OfficialTrack1Row],
    config: Dict[str, Any],
    progress: bool = True,
) -> Tuple[List[OfficialTrack1Row], List[Dict[str, Any]]]:
    """Repair isolated jump-return and z outliers without dropping rows."""
    rules = config.get("outlier_repair", {})
    if not bool(rules.get("enabled", True)):
        return list(rows), []
    grouped = group_tracks(rows)
    output = []
    changes = []
    minimum = int(rules.get("min_track_length", 5))
    ratio = float(rules.get("max_repairs_per_track_ratio", 0.10))
    max_gap = int(rules.get("max_gap_for_interpolation", 10))
    for key, track in progress_iter(list(grouped.items()), progress, "V4 outlier repair"):
        if len(track) < minimum:
            output.extend(track)
            continue
        points = np.asarray([position(row) for row in track], dtype=float)
        frames = [int(row.frame_id) for row in track]
        refined = points.copy()
        median_steps = _median_step_per_frame(points, frames)
        absolute = _class_threshold(rules, int(key[1]))
        dynamic = max(absolute, median_steps * float(rules.get("step_multiplier_vs_median", 4.0)))
        max_repairs = max(1, int(np.floor(float(len(track)) * ratio)))
        candidates = []
        track_z_median = float(np.median(points[:, 2]))
        for index in range(1, len(track) - 1):
            gap_before = frames[index] - frames[index - 1]
            gap_after = frames[index + 1] - frames[index]
            if gap_before <= 0 or gap_after <= 0 or gap_before > max_gap or gap_after > max_gap:
                continue
            total_gap = frames[index + 1] - frames[index - 1]
            weight = float(frames[index] - frames[index - 1]) / float(max(1, total_gap))
            prediction = points[index - 1] * (1.0 - weight) + points[index + 1] * weight
            residual = float(np.linalg.norm(points[index] - prediction))
            before = float(np.linalg.norm(points[index] - points[index - 1])) / float(gap_before)
            after = float(np.linalg.norm(points[index + 1] - points[index])) / float(gap_after)
            neighbor_motion = float(np.linalg.norm(points[index + 1] - points[index - 1])) / float(max(1, total_gap))
            jump_return = before > dynamic and after > dynamic and neighbor_motion <= max(dynamic * 0.5, median_steps * 2.0)
            z_outlier = abs(float(points[index, 2]) - track_z_median) > float(rules.get("z_outlier_threshold_m", 5.0)) and abs(float(points[index - 1, 2] - points[index + 1, 2])) <= float(rules.get("z_outlier_threshold_m", 5.0))
            if (jump_return and residual > dynamic) or z_outlier:
                candidates.append((max(residual, abs(float(points[index, 2]) - track_z_median)), index, prediction, "jump_return" if jump_return else "z_outlier"))
        candidates = sorted(candidates, key=lambda item: item[0], reverse=True)[:max_repairs]
        for _score, index, target, reason in candidates:
            refined[index] = target
            distance = float(np.linalg.norm(target - points[index]))
            changes.append({
                "scene_id": key[0], "class_id": key[1], "object_id": key[2], "frame_id": track[index].frame_id,
                "field_group": "position", "reason": reason, "old_x": float(points[index, 0]),
                "old_y": float(points[index, 1]), "old_z": float(points[index, 2]), "new_x": float(target[0]),
                "new_y": float(target[1]), "new_z": float(target[2]), "position_change_m": distance,
            })
        for index, row in enumerate(track):
            output.append(clone_row(row, x=float(refined[index, 0]), y=float(refined[index, 1]), z=float(refined[index, 2])))
    return sorted(output, key=lambda row: row.key()), changes


def _median_step_per_frame(points: np.ndarray, frames: Sequence[int]) -> float:
    values = []
    for index in range(1, len(points)):
        gap = max(1, int(frames[index]) - int(frames[index - 1]))
        values.append(float(np.linalg.norm(points[index] - points[index - 1])) / float(gap))
    return float(np.median(values)) if values else 0.0


def _class_threshold(rules: Dict[str, Any], class_id: int) -> float:
    values = rules.get("absolute_step_threshold_by_class", {})
    return float(values.get(class_id, values.get(str(class_id), rules.get("absolute_step_threshold_m_default", 20.0))))

