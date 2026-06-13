"""Geometry, smoothness and V3.1 change metrics for V4 variants."""

from collections import defaultdict
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from deep_oc_sort_3d.official_023_027.official_track1_io import OfficialTrack1Row
from deep_oc_sort_3d.v4_geometry_refinement.track1_geometry_io import dimensions, group_tracks, position, unique_track_count
from deep_oc_sort_3d.v4_geometry_refinement.yaw_refinement import angle_delta


def compute_geometry_metrics(
    rows: Sequence[OfficialTrack1Row],
    config: Dict[str, Any],
    baseline_rows: Sequence[OfficialTrack1Row] = (),
    stage_changes: Sequence[Dict[str, Any]] = (),
) -> Dict[str, Any]:
    """Compute mandatory global and per-scene/class geometry metrics."""
    grouped = group_tracks(rows)
    rules = config.get("geometry_metrics", {})
    step_values = []
    step_xy_values = []
    track_lengths = []
    suspect_tracks = []
    suspect_points = []
    z_outliers = 0
    yaw_jumps = 0
    yaw_steps = 0
    jump_count = 0
    dimension_variances = []
    step_by_scene = defaultdict(list)
    step_by_class = defaultdict(list)
    suspect_by_scene = defaultdict(int)
    suspect_by_class = defaultdict(int)
    for key, track in grouped.items():
        track_lengths.append(len(track))
        points = np.asarray([position(row) for row in track], dtype=float)
        dims = np.asarray([dimensions(row) for row in track], dtype=float)
        frames = [int(row.frame_id) for row in track]
        class_id = int(key[1])
        threshold = _class_step_threshold(rules, class_id)
        track_suspect = False
        for index in range(1, len(track)):
            gap = max(1, frames[index] - frames[index - 1])
            distance = float(np.linalg.norm(points[index] - points[index - 1])) / float(gap)
            distance_xy = float(np.linalg.norm(points[index, :2] - points[index - 1, :2])) / float(gap)
            step_values.append(distance)
            step_xy_values.append(distance_xy)
            step_by_scene[int(key[0])].append(distance)
            step_by_class[class_id].append(distance)
            if distance > threshold:
                track_suspect = True
                jump_count += 1
                suspect_points.append({
                    "scene_id": key[0], "class_id": class_id, "object_id": key[2], "frame_id": track[index].frame_id,
                    "reason": "step_distance", "value": distance, "threshold": threshold,
                })
            yaw_steps += 1
            yaw_delta = abs(angle_delta(float(track[index - 1].yaw), float(track[index].yaw))) / float(gap)
            if yaw_delta > float(rules.get("yaw_jump_threshold_rad", 1.0)):
                yaw_jumps += 1
        z_median = float(np.median(points[:, 2]))
        z_threshold = float(rules.get("z_outlier_threshold_m", 5.0))
        for index, row in enumerate(track):
            deviation = abs(float(points[index, 2]) - z_median)
            if deviation > z_threshold:
                z_outliers += 1
                track_suspect = True
                suspect_points.append({
                    "scene_id": key[0], "class_id": class_id, "object_id": key[2], "frame_id": row.frame_id,
                    "reason": "z_outlier", "value": deviation, "threshold": z_threshold,
                })
        dimension_variances.append(float(np.mean(np.var(dims, axis=0))))
        if track_suspect:
            suspect_tracks.append({"scene_id": key[0], "class_id": class_id, "object_id": key[2], "length": len(track)})
            suspect_by_scene[int(key[0])] += 1
            suspect_by_class[class_id] += 1
    changes = compare_geometry_rows(baseline_rows, rows) if baseline_rows else []
    position_changes = [float(row["position_change_m"]) for row in changes if row.get("position_change_m") is not None]
    dimension_changes = [row for row in changes if row.get("dimension_changed")]
    yaw_changes = [row for row in changes if row.get("yaw_changed")]
    dimension_change_values = [float(row.get("max_dimension_change", 0.0)) for row in dimension_changes]
    dimension_change_ratios = [float(row.get("max_dimension_change_ratio", 0.0)) for row in dimension_changes]
    yaw_change_values = [float(row.get("yaw_change_rad", 0.0)) for row in yaw_changes]
    repaired_keys = set((row.get("scene_id"), row.get("class_id"), row.get("object_id"), row.get("frame_id")) for row in stage_changes if row.get("stage") == "outlier_repair")
    repaired_tracks = set(key[:3] for key in repaired_keys)
    repair_scene = defaultdict(int)
    repair_class = defaultdict(int)
    for key in repaired_keys:
        repair_scene[int(key[0])] += 1
        repair_class[int(key[1])] += 1
    dimension_class = defaultdict(int)
    yaw_class = defaultdict(int)
    for row in dimension_changes:
        dimension_class[int(row["class_id"])] += 1
    for row in yaw_changes:
        yaw_class[int(row["class_id"])] += 1
    summary = {
        "rows": len(rows), "unique_tracks": unique_track_count(rows), "track_count": len(grouped),
        "rows_per_track_mean": _mean(track_lengths), "rows_per_track_median": _pct(track_lengths, 50),
        "mean_track_length": _mean(track_lengths), "median_track_length": _pct(track_lengths, 50),
        "step_mean": _mean(step_values), "step_median": _pct(step_values, 50), "step_p90": _pct(step_values, 90),
        "step_p95": _pct(step_values, 95), "step_p99": _pct(step_values, 99), "step_max": max(step_values) if step_values else None,
        "step_xy_mean": _mean(step_xy_values), "step_xy_median": _pct(step_xy_values, 50),
        "step_xy_p90": _pct(step_xy_values, 90), "step_xy_p95": _pct(step_xy_values, 95),
        "step_xy_p99": _pct(step_xy_values, 99), "step_xy_max": max(step_xy_values) if step_xy_values else None,
        "suspect_track_count": len(suspect_tracks), "suspect_track_ratio": _ratio(len(suspect_tracks), len(grouped)),
        "suspect_point_count": len(suspect_points), "suspect_point_ratio": _ratio(len(suspect_points), len(rows)),
        "jump_count": jump_count, "jump_ratio": _ratio(jump_count, len(step_values)),
        "z_outlier_count": z_outliers, "z_outlier_ratio": _ratio(z_outliers, len(rows)),
        "tracks_with_repairs": len(repaired_tracks), "points_repaired": len(repaired_keys),
        "points_changed": len(changes), "mean_position_change_m": _mean(position_changes),
        "p95_position_change_m": _pct(position_changes, 95), "max_position_change_m": max(position_changes) if position_changes else 0.0,
        "dimension_variance_mean": _mean(dimension_variances), "dimension_variance_median": _pct(dimension_variances, 50),
        "dimension_change_count": len(dimension_changes),
        "dimension_change_mean": _mean(dimension_change_values), "dimension_change_p95": _pct(dimension_change_values, 95),
        "dimension_change_max": max(dimension_change_values) if dimension_change_values else 0.0,
        "dimension_change_ratio_mean": _mean(dimension_change_ratios),
        "dimension_change_ratio_p95": _pct(dimension_change_ratios, 95),
        "dimension_change_ratio_max": max(dimension_change_ratios) if dimension_change_ratios else 0.0,
        "yaw_jump_count": yaw_jumps, "yaw_jump_ratio": _ratio(yaw_jumps, yaw_steps),
        "yaw_changed_count": len(yaw_changes), "yaw_change_mean_rad": _mean(yaw_change_values),
        "yaw_change_p95_rad": _pct(yaw_change_values, 95),
        "yaw_change_max_rad": max(yaw_change_values) if yaw_change_values else 0.0,
        "scene_distribution": _row_distribution(rows, "scene_id"), "class_distribution": _row_distribution(rows, "class_id"),
        "step_p95_by_scene": _percentile_mapping(step_by_scene, 95), "step_p95_by_class": _percentile_mapping(step_by_class, 95),
        "suspect_tracks_by_scene": dict(sorted(suspect_by_scene.items())), "suspect_tracks_by_class": dict(sorted(suspect_by_class.items())),
        "points_repaired_by_scene": dict(sorted(repair_scene.items())), "points_repaired_by_class": dict(sorted(repair_class.items())),
        "dimension_changes_by_class": dict(sorted(dimension_class.items())), "yaw_changes_by_class": dict(sorted(yaw_class.items())),
        "suspect_tracks": suspect_tracks, "suspect_points": suspect_points, "changed_points": changes,
    }
    return summary


def compare_geometry_rows(baseline: Sequence[OfficialTrack1Row], refined: Sequence[OfficialTrack1Row]) -> List[Dict[str, Any]]:
    """Describe final per-row geometry changes relative to immutable V3.1 keys."""
    baseline_map = {row.key(): row for row in baseline}
    output = []
    for row in refined:
        old = baseline_map.get(row.key())
        if old is None:
            continue
        position_change = float(np.linalg.norm(position(row) - position(old)))
        old_dimensions = dimensions(old)
        dimension_change = float(np.max(np.abs(dimensions(row) - old_dimensions)))
        dimension_ratio = float(np.max(np.abs(dimensions(row) - old_dimensions) / np.maximum(np.abs(old_dimensions), 1e-6)))
        yaw_change = abs(angle_delta(float(old.yaw), float(row.yaw)))
        if position_change <= 1e-9 and dimension_change <= 1e-9 and yaw_change <= 1e-9:
            continue
        output.append({
            "scene_id": row.scene_id, "class_id": row.class_id, "object_id": row.object_id, "frame_id": row.frame_id,
            "position_change_m": position_change if position_change > 1e-9 else None,
            "dimension_changed": dimension_change > 1e-9, "max_dimension_change": dimension_change,
            "max_dimension_change_ratio": dimension_ratio,
            "yaw_changed": yaw_change > 1e-9, "yaw_change_rad": yaw_change,
        })
    return output


def compact_metrics(summary: Dict[str, Any]) -> Dict[str, Any]:
    """Remove verbose point lists for JSON/CSV comparison outputs."""
    return {key: value for key, value in summary.items() if key not in ("suspect_tracks", "suspect_points", "changed_points")}


def _class_step_threshold(rules: Dict[str, Any], class_id: int) -> float:
    values = rules.get("suspicious_step_m_by_class", {})
    return float(values.get(class_id, values.get(str(class_id), rules.get("suspicious_step_m_default", 12.0))))


def _row_distribution(rows: Sequence[OfficialTrack1Row], field: str) -> Dict[str, int]:
    counts = defaultdict(int)
    for row in rows:
        counts[str(getattr(row, field))] += 1
    return dict(sorted(counts.items(), key=lambda item: int(item[0])))


def _percentile_mapping(values: Dict[int, List[float]], percentile: float) -> Dict[str, Any]:
    return {str(key): _pct(items, percentile) for key, items in sorted(values.items())}


def _mean(values: Sequence[float]) -> Any:
    return float(np.mean(values)) if values else None


def _pct(values: Sequence[float], percentile: float) -> Any:
    return float(np.percentile(values, percentile)) if values else None


def _ratio(numerator: int, denominator: int) -> Any:
    return float(numerator) / float(denominator) if denominator else None
