"""Train/val calibration and test-change metrics for Step 22D."""

from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from deep_oc_sort_3d.official_023_027.official_track1_io import OfficialTrack1Row
from deep_oc_sort_3d.v5_geometry_calibration.geometry_calibration_io import angle_delta, dimensions, position, unique_track_count


def summarize_match_rows(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Summarize GT-prediction error columns."""
    center = _values(rows, "center_error_before")
    dimension = _values(rows, "dimension_error_before")
    yaw = _values(rows, "yaw_error_before")
    depth = _values(rows, "depth_error_before")
    iou = _values(rows, "iou3d_proxy_before")
    return {
        "samples": len(rows),
        "center_error_mean": _mean(center), "center_error_median": _pct(center, 50),
        "center_error_p75": _pct(center, 75), "center_error_p90": _pct(center, 90), "center_error_p95": _pct(center, 95),
        "dimension_error_mean": _mean(dimension), "dimension_error_median": _pct(dimension, 50),
        "dimension_error_p90": _pct(dimension, 90),
        "yaw_error_mean": _mean(yaw), "yaw_error_median": _pct(yaw, 50), "yaw_error_p90": _pct(yaw, 90),
        "depth_error_mean": _mean(depth), "depth_error_median": _pct(depth, 50),
        "3d_iou_proxy_mean": _mean(iou), "3d_iou_proxy_median": _pct(iou, 50),
    }


def evaluate_corrections(rows: Sequence[Dict[str, Any]], corrections: Dict[str, Any]) -> Dict[str, Any]:
    """Evaluate selected class-level corrections against calibration GT."""
    evaluated: List[Dict[str, Any]] = []
    for row in rows:
        class_id = str(int(float(row["official_class_id"])))
        pred_center = np.asarray([row["pred_x"], row["pred_y"], row["pred_z"]], dtype=float)
        pred_dims = np.asarray([row["pred_width"], row["pred_length"], row["pred_height"]], dtype=float)
        pred_yaw = float(row["pred_yaw"])
        gt_center = np.asarray([row["gt_x"], row["gt_y"], row["gt_z"]], dtype=float)
        gt_dims = np.asarray([row["gt_width"], row["gt_length"], row["gt_height"]], dtype=float)
        gt_yaw = float(row["gt_yaw"])
        dimension_item = corrections.get("dimension", {}).get(class_id, {})
        center_item = corrections.get("center", {}).get(class_id, {})
        yaw_item = corrections.get("yaw", {}).get(class_id, {})
        if dimension_item.get("selected"):
            pred_dims = pred_dims * np.asarray(dimension_item.get("scale", [1.0, 1.0, 1.0]), dtype=float)
        if center_item.get("selected"):
            pred_center = pred_center + np.asarray(center_item.get("bias", [0.0, 0.0, 0.0]), dtype=float)
        if yaw_item.get("selected"):
            pred_yaw = pred_yaw + float(yaw_item.get("bias_rad", 0.0))
        evaluated.append({
            "center_error_before": float(np.linalg.norm(pred_center - gt_center)),
            "dimension_error_before": float(np.mean(np.abs(pred_dims - gt_dims))),
            "yaw_error_before": abs(angle_delta(pred_yaw, gt_yaw)),
            "depth_error_before": abs(float(np.linalg.norm(pred_center)) - float(np.linalg.norm(gt_center))),
            "iou3d_proxy_before": axis_aligned_iou3d(pred_center, pred_dims, gt_center, gt_dims),
        })
    return summarize_match_rows(evaluated)


def before_after_rows(rows: Sequence[Dict[str, Any]], corrections_by_name: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Return flat before/after metrics for diagnostic CSVs."""
    before = summarize_match_rows(rows)
    output: List[Dict[str, Any]] = []
    for name, corrections in corrections_by_name.items():
        after = evaluate_corrections(rows, corrections)
        for metric in sorted(before.keys()):
            if metric == "samples":
                continue
            before_value = before.get(metric)
            after_value = after.get(metric)
            output.append({
                "variant": name, "metric": metric, "before": before_value, "after": after_value,
                "delta": _delta(after_value, before_value), "samples": len(rows),
            })
    return output


def summarize_test_changes(
    baseline: Sequence[OfficialTrack1Row],
    candidate: Sequence[OfficialTrack1Row],
) -> Dict[str, Any]:
    """Measure V5 geometry changes while preserving immutable keys."""
    baseline_map = {row.key(): row for row in baseline}
    position_changes = []
    dimension_ratios = []
    yaw_changes = []
    scene_counts = defaultdict(int)
    class_counts = defaultdict(int)
    for row in candidate:
        scene_counts[str(row.scene_id)] += 1
        class_counts[str(row.class_id)] += 1
        old = baseline_map.get(row.key())
        if old is None:
            continue
        position_changes.append(float(np.linalg.norm(position(row) - position(old))))
        old_dims = dimensions(old)
        dimension_ratios.append(float(np.max(np.abs(dimensions(row) - old_dims) / np.maximum(np.abs(old_dims), 1e-6))))
        yaw_changes.append(abs(angle_delta(float(old.yaw), float(row.yaw))))
    return {
        "rows": len(candidate), "unique_tracks": unique_track_count(candidate),
        "scene_distribution": dict(sorted(scene_counts.items(), key=lambda item: int(item[0]))),
        "class_distribution": dict(sorted(class_counts.items(), key=lambda item: int(item[0]))),
        "mean_position_change_m": _mean(position_changes), "p95_position_change_m": _pct(position_changes, 95),
        "max_position_change_m": max(position_changes) if position_changes else 0.0,
        "mean_dimension_change_ratio": _mean(dimension_ratios), "p95_dimension_change_ratio": _pct(dimension_ratios, 95),
        "max_dimension_change_ratio": max(dimension_ratios) if dimension_ratios else 0.0,
        "yaw_changed_count": sum(1 for value in yaw_changes if value > 1e-9),
        "yaw_change_mean": _mean(yaw_changes), "yaw_change_p95": _pct(yaw_changes, 95),
    }


def axis_aligned_iou3d(center_a: np.ndarray, dims_a: np.ndarray, center_b: np.ndarray, dims_b: np.ndarray) -> Optional[float]:
    """Compute an axis-aligned 3D IoU proxy, ignoring yaw."""
    if np.any(dims_a <= 0.0) or np.any(dims_b <= 0.0):
        return None
    min_a, max_a = center_a - dims_a / 2.0, center_a + dims_a / 2.0
    min_b, max_b = center_b - dims_b / 2.0, center_b + dims_b / 2.0
    intersection = np.maximum(0.0, np.minimum(max_a, max_b) - np.maximum(min_a, min_b))
    intersection_volume = float(np.prod(intersection))
    union = float(np.prod(dims_a) + np.prod(dims_b) - intersection_volume)
    return None if union <= 0.0 else intersection_volume / union


def _values(rows: Sequence[Dict[str, Any]], key: str) -> List[float]:
    output = []
    for row in rows:
        try:
            value = float(row[key])
            if np.isfinite(value):
                output.append(value)
        except (KeyError, TypeError, ValueError):
            continue
    return output


def _mean(values: Sequence[float]) -> Optional[float]:
    return float(np.mean(values)) if values else None


def _pct(values: Sequence[float], percentile: float) -> Optional[float]:
    return float(np.percentile(values, percentile)) if values else None


def _delta(after: Any, before: Any) -> Optional[float]:
    try:
        return float(after) - float(before)
    except (TypeError, ValueError):
        return None
