"""Audit motion-filter rejections, gaps and pseudo-3D jump proxies."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from deep_oc_sort_3d.bytetrack_audit.audit_config import VARIANT_NAMES, audit_scenes, output_root, variant_paths
from deep_oc_sort_3d.bytetrack_audit.audit_io import (
    CsvSink,
    iter_csv,
    iter_jsonl,
    progress_iter,
    safe_bool,
    safe_float,
    safe_int,
    write_csv,
    write_json,
)


DETAIL_FIELDS = [
    "variant_name", "subset", "scene_name", "class_id", "class_name", "candidate_id",
    "tracklet_id", "camera_id", "camera_pair", "num_records", "duration_frames", "num_gaps",
    "max_gap", "mean_gap", "median_gap", "gap_bucket", "step_p95", "step_p99", "step_max",
    "jump_count", "jump_ratio", "motion_status_before", "motion_status_after", "rejection_reason",
    "center_3d_delta", "xy_delta", "z_delta", "depth_delta", "bbox_height_delta",
    "bbox_height_ratio", "gap_length_at_jump", "pseudo3d_source", "projection_valid",
]


def run_motion_filter_audit(config: Dict[str, Any], progress: bool = True) -> Dict[str, Any]:
    """Audit all motion metrics for V2 current, 21B and 21C best."""
    if not bool(config.get("motion_filter_audit", {}).get("enabled", True)):
        return {"status": "disabled"}
    root = output_root(config) / "motion_filter_audit"
    allowed = set((subset, scene) for subset, _split, scene in audit_scenes(config, include_test=True))
    aggregates = _new_aggregates()
    summary_rows = []
    velocity_values = {}
    with CsvSink(root / "rejected_candidates.csv", DETAIL_FIELDS) as rejected_sink, CsvSink(
        root / "accepted_candidates.csv", DETAIL_FIELDS
    ) as accepted_sink, CsvSink(root / "pseudo3d_jump_analysis.csv", DETAIL_FIELDS) as jump_sink, CsvSink(
        root / "bbox_height_delta_gap_analysis.csv", DETAIL_FIELDS
    ) as bbox_sink:
        for variant_name in VARIANT_NAMES:
            paths = variant_paths(config, variant_name)
            motion_root = paths.get("motion_clean_root", Path(""))
            candidate_root = paths.get("candidates_root", Path(""))
            metric_files = _selected_metric_files(motion_root, allowed)
            variant_total = 0
            variant_rejected = 0
            for metric_path in progress_iter(metric_files, progress, "%s motion audit" % variant_name):
                subset, scene_name, camera_id = _identity(motion_root, metric_path)
                candidate_lookup = _candidate_lookup(candidate_root, subset, scene_name, camera_id)
                for metric in iter_csv(metric_path):
                    candidate = candidate_lookup.get(str(metric.get("candidate_id", "")), {})
                    row = _detail_row(variant_name, subset, scene_name, camera_id, metric, candidate, config)
                    clean = safe_bool(metric.get("is_motion_clean"))
                    variant_total += 1
                    if clean:
                        accepted_sink.write(row)
                    else:
                        rejected_sink.write(row)
                        variant_rejected += 1
                    jump_sink.write(row)
                    bbox_sink.write(row)
                    _update_aggregates(aggregates, row, clean)
                    value = safe_float(row.get("step_p95"))
                    if value is not None:
                        velocity_values.setdefault((variant_name, row.get("class_name")), []).append(value)
            summary_rows.append(
                {
                    "variant_name": variant_name,
                    "total_candidates": variant_total,
                    "accepted_candidates": variant_total - variant_rejected,
                    "rejected_candidates": variant_rejected,
                    "rejection_rate": None if variant_total <= 0 else float(variant_rejected) / float(variant_total),
                }
            )
    reason_rows = _aggregate_rows(aggregates["reason"], "rejection_reason")
    gap_rows = _aggregate_rows(aggregates["gap"], "gap_bucket")
    class_rows = _aggregate_rows(aggregates["class"], "class_name")
    scene_rows = _aggregate_rows(aggregates["scene"], "scene_name")
    camera_rows = _aggregate_rows(aggregates["camera"], "camera_pair")
    length_rows = _aggregate_rows(aggregates["length"], "track_length_bucket")
    bbox_rows = _aggregate_rows(aggregates["bbox"], "bbox_height_delta_bucket")
    velocity_rows = _velocity_rows(velocity_values)
    write_csv(root / "motion_filter_summary.csv", summary_rows)
    write_json(root / "motion_filter_summary.json", {"status": "ok", "rows": summary_rows})
    write_csv(root / "rejected_candidate_reason_summary.csv", reason_rows)
    write_csv(root / "rejected_by_gap_bucket.csv", gap_rows)
    write_csv(root / "rejected_by_class.csv", class_rows)
    write_csv(root / "rejected_by_scene.csv", scene_rows)
    write_csv(root / "rejected_by_camera_pair.csv", camera_rows)
    write_csv(root / "rejected_by_track_length.csv", length_rows)
    write_csv(root / "rejected_by_bbox_height_delta.csv", bbox_rows)
    write_csv(root / "velocity_percentile_diagnostics.csv", velocity_rows)
    return {
        "status": "ok",
        "summary_rows": summary_rows,
        "reason_rows": reason_rows,
        "gap_rows": gap_rows,
        "class_rows": class_rows,
        "scene_rows": scene_rows,
        "camera_rows": camera_rows,
        "length_rows": length_rows,
        "bbox_rows": bbox_rows,
    }


def _detail_row(
    variant: str,
    subset: str,
    scene: str,
    camera: str,
    metric: Dict[str, Any],
    candidate: Dict[str, Any],
    config: Dict[str, Any],
) -> Dict[str, Any]:
    steps = _json_list(metric.get("step_distances_3d_json"))
    gaps = [safe_int(item[3], 1) for item in steps if isinstance(item, list) and len(item) >= 4]
    distances = [float(item[2]) for item in steps if isinstance(item, list) and len(item) >= 3]
    trajectory_2d = candidate.get("trajectory_2d_sampled", candidate.get("trajectory_2d", [])) or []
    trajectory_3d = candidate.get("trajectory_3d_sampled", candidate.get("trajectory_3d", [])) or []
    bbox_delta, bbox_ratio, bbox_gap = _bbox_height_jump(trajectory_2d)
    center_delta, xy_delta, z_delta, center_gap = _center_jump(trajectory_3d)
    max_gap = max(gaps) if gaps else max(bbox_gap, center_gap, 0)
    clean = safe_bool(metric.get("is_motion_clean"))
    return {
        "variant_name": variant,
        "subset": subset,
        "scene_name": scene,
        "class_id": safe_int(metric.get("class_id"), safe_int(candidate.get("class_id"), -1)),
        "class_name": metric.get("class_name", candidate.get("class_name", "")),
        "candidate_id": metric.get("candidate_id", candidate.get("candidate_id", "")),
        "tracklet_id": "%s__%s__track_%s" % (scene, camera, metric.get("local_track_id", "")),
        "camera_id": camera,
        "camera_pair": camera,
        "num_records": safe_int(metric.get("length"), safe_int(candidate.get("length"), 0)),
        "duration_frames": safe_int(candidate.get("duration"), safe_int(metric.get("length"), 0)),
        "num_gaps": sum(1 for gap in gaps if gap > 1),
        "max_gap": max_gap,
        "mean_gap": _mean(gaps),
        "median_gap": _percentile(gaps, 50),
        "gap_bucket": _gap_bucket(max_gap, config),
        "step_p95": safe_float(metric.get("p95_step_distance_3d")),
        "step_p99": _percentile(distances, 99),
        "step_max": safe_float(metric.get("max_step_distance_3d")),
        "jump_count": safe_int(metric.get("jump_count"), 0),
        "jump_ratio": safe_float(metric.get("jump_ratio"), 0.0),
        "motion_status_before": candidate.get("quality_flag", "candidate"),
        "motion_status_after": metric.get("motion_quality_flag", ""),
        "rejection_reason": "ok" if clean else str(metric.get("motion_reject_reason", "unknown")),
        "center_3d_delta": center_delta,
        "xy_delta": xy_delta,
        "z_delta": z_delta,
        "depth_delta": "",
        "bbox_height_delta": bbox_delta,
        "bbox_height_ratio": bbox_ratio,
        "gap_length_at_jump": max(bbox_gap, center_gap),
        "pseudo3d_source": candidate.get("pseudo3d_source", candidate.get("source", "unknown")),
        "projection_valid": candidate.get("projection_valid", ""),
    }


def _candidate_lookup(root: Path, subset: str, scene: str, camera: str) -> Dict[str, Dict[str, Any]]:
    path = root / subset / scene / (camera + "_candidates.jsonl")
    if not path.exists():
        path = path.with_suffix(".csv")
    iterator = iter_jsonl(path) if path.suffix == ".jsonl" else iter_csv(path)
    return {str(row.get("candidate_id", "")): row for row in iterator}


def _selected_metric_files(root: Path, allowed: set) -> List[Path]:
    if not root.exists():
        return []
    output = []
    for path in sorted(root.rglob("*_motion_metrics.csv")):
        subset, scene, _camera = _identity(root, path)
        if (subset, scene) in allowed:
            output.append(path)
    return output


def _identity(root: Path, path: Path) -> Tuple[str, str, str]:
    relative = path.relative_to(root)
    subset = relative.parts[0] if len(relative.parts) >= 3 else ""
    scene = relative.parts[1] if len(relative.parts) >= 3 else ""
    camera = path.stem.replace("_motion_metrics", "")
    return subset, scene, camera


def _bbox_height_jump(trajectory: Any) -> Tuple[Optional[float], Optional[float], int]:
    best_delta = None
    best_ratio = None
    best_gap = 0
    previous = None
    for item in (trajectory if isinstance(trajectory, list) else []):
        if not isinstance(item, list) or len(item) < 5:
            continue
        frame, height = safe_int(item[0], 0), abs(float(item[4]) - float(item[2]))
        if previous is not None:
            delta = abs(height - previous[1])
            ratio = max(height, previous[1]) / max(1e-6, min(height, previous[1]))
            if best_delta is None or delta > best_delta:
                best_delta, best_ratio, best_gap = delta, ratio, frame - previous[0]
        previous = (frame, height)
    return best_delta, best_ratio, best_gap


def _center_jump(trajectory: Any) -> Tuple[Optional[float], Optional[float], Optional[float], int]:
    best = None
    best_xy = None
    best_z = None
    best_gap = 0
    previous = None
    for item in (trajectory if isinstance(trajectory, list) else []):
        if not isinstance(item, list) or len(item) < 4:
            continue
        current = (safe_int(item[0], 0), float(item[1]), float(item[2]), float(item[3]))
        if previous is not None:
            dx, dy, dz = current[1] - previous[1], current[2] - previous[2], current[3] - previous[3]
            distance = float(np.sqrt(dx * dx + dy * dy + dz * dz))
            if best is None or distance > best:
                best = distance
                best_xy = float(np.sqrt(dx * dx + dy * dy))
                best_z = abs(dz)
                best_gap = current[0] - previous[0]
        previous = current
    return best, best_xy, best_z, best_gap


def _json_list(value: Any) -> List[Any]:
    if isinstance(value, list):
        return value
    if value in (None, ""):
        return []
    try:
        parsed = json.loads(str(value))
    except ValueError:
        return []
    return parsed if isinstance(parsed, list) else []


def _gap_bucket(value: int, config: Dict[str, Any]) -> str:
    if value <= 1:
        return "gap_0_or_1"
    if value <= 5:
        return "gap_2_5"
    if value <= 15:
        return "gap_6_15"
    if value <= 30:
        return "gap_16_30"
    return "gap_gt_30"


def _length_bucket(value: int) -> str:
    if value <= 3:
        return "length_1_3"
    if value <= 10:
        return "length_4_10"
    if value <= 50:
        return "length_11_50"
    return "length_gt_50"


def _bbox_bucket(value: Any) -> str:
    number = safe_float(value)
    if number is None:
        return "unknown"
    if number < 5:
        return "delta_lt_5"
    if number < 20:
        return "delta_5_20"
    if number < 50:
        return "delta_20_50"
    return "delta_ge_50"


def _new_aggregates() -> Dict[str, Dict[Any, Dict[str, int]]]:
    return {key: {} for key in ["reason", "gap", "class", "scene", "camera", "length", "bbox"]}


def _update_aggregates(values: Dict[str, Any], row: Dict[str, Any], clean: bool) -> None:
    keys = {
        "reason": row.get("rejection_reason"),
        "gap": row.get("gap_bucket"),
        "class": row.get("class_name"),
        "scene": row.get("scene_name"),
        "camera": row.get("camera_pair"),
        "length": _length_bucket(safe_int(row.get("num_records"), 0)),
        "bbox": _bbox_bucket(row.get("bbox_height_delta")),
    }
    for name, key in keys.items():
        aggregate_key = (row.get("variant_name"), key)
        target = values[name].setdefault(aggregate_key, {"total": 0, "rejected": 0})
        target["total"] += 1
        target["rejected"] += int(not clean)


def _aggregate_rows(values: Dict[Any, Dict[str, int]], key_name: str) -> List[Dict[str, Any]]:
    output = []
    for (variant, key), counts in sorted(values.items(), key=lambda item: str(item[0])):
        total = counts["total"]
        rejected = counts["rejected"]
        output.append(
            {
                "variant_name": variant,
                key_name: key,
                "total_candidates": total,
                "rejected_candidates": rejected,
                "accepted_candidates": total - rejected,
                "rejection_rate": None if total <= 0 else float(rejected) / float(total),
            }
        )
    return output


def _velocity_rows(values: Dict[Any, List[float]]) -> List[Dict[str, Any]]:
    output = []
    for (variant, class_name), numbers in sorted(values.items()):
        output.append(
            {
                "variant_name": variant,
                "class_name": class_name,
                "count": len(numbers),
                "p50_step_p95": _percentile(numbers, 50),
                "p75_step_p95": _percentile(numbers, 75),
                "p90_step_p95": _percentile(numbers, 90),
                "p95_step_p95": _percentile(numbers, 95),
                "p99_step_p95": _percentile(numbers, 99),
            }
        )
    return output


def _mean(values: List[Any]) -> Optional[float]:
    return None if not values else float(np.mean(np.asarray(values, dtype=np.float64)))


def _percentile(values: List[Any], percentile: float) -> Optional[float]:
    return None if not values else float(np.percentile(np.asarray(values, dtype=np.float64), percentile))
