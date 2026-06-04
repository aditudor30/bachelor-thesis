"""Coverage, projection, and smoothness audit for Step 15G outputs."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from deep_oc_sort_3d.audit3d.audit3d_io import progress_iter, write_csv, write_json
from deep_oc_sort_3d.pseudo3d.pseudo3d_io import load_scene_camera_calibration
from deep_oc_sort_3d.pseudo3d.pseudo3d_projection_check import check_prediction_projection, summarize_projection_checks
from deep_oc_sort_3d.pseudo3d.pseudo3d_smoothness import audit_pseudo3d_smoothness, worst_pseudo3d_jumps
from deep_oc_sort_3d.pseudo3d_fullcam.fullcam_discovery import FullCamItem


def audit_fullcam_coverage(
    items: List[FullCamItem],
    output_root: Path,
    config: Optional[Dict[str, Any]] = None,
    show_progress: bool = True,
) -> Dict[str, Any]:
    """Audit generated full-camera pseudo-3D outputs."""
    cfg = config or {}
    coverage_cfg = cfg.get("coverage", {})
    check_projection = bool(coverage_cfg.get("check_projection", True))
    check_smoothness = bool(coverage_cfg.get("check_smoothness", True))
    per_camera = []
    projection_rows = []
    smoothness_rows = []
    worst_jumps = []
    for item in progress_iter(items, show_progress, "audit fullcam pseudo3D coverage", "camera"):
        raw_stats = _prediction_stats(Path(item.raw_prediction_path))
        stabilized_stats = _prediction_stats(Path(item.stabilized_prediction_path))
        row = _camera_row(item, raw_stats, stabilized_stats)
        per_camera.append(row)
        if check_projection and Path(item.stabilized_prediction_path).exists():
            projection_rows.extend(_projection_rows_for_item(item, cfg))
        if check_smoothness and Path(item.stabilized_prediction_path).exists():
            smoothness_row, jumps = _smoothness_for_item(item, cfg)
            smoothness_rows.append(smoothness_row)
            worst_jumps.extend(jumps)
    summary = _summary_from_rows(items, per_camera, projection_rows, smoothness_rows)
    summary["per_camera"] = per_camera
    summary["projection_rows"] = projection_rows
    summary["projection"] = summarize_projection_checks(projection_rows) if projection_rows else {"projection_success_rate": None, "total": 0}
    summary["smoothness"] = _aggregate_smoothness(smoothness_rows)
    summary["smoothness_rows"] = smoothness_rows
    summary["worst_remaining_jumps"] = sorted(worst_jumps, key=lambda row: float(row.get("step_distance", 0.0) or 0.0), reverse=True)[:100]
    return summary


def write_coverage_reports(summary: Dict[str, Any], output_root: Path) -> None:
    """Write Step 15G coverage reports under the requested output structure."""
    coverage_root = output_root / "coverage_audit"
    projection_root = output_root / "projection_checks"
    smoothness_root = output_root / "smoothness"
    summaries_root = output_root / "summaries"
    evaluation_root = output_root / "evaluation"
    compact = _compact_summary(summary)
    write_json(compact, coverage_root / "pseudo3d_fullcam_coverage_summary.json")
    write_csv(_summary_rows(compact), coverage_root / "pseudo3d_fullcam_coverage_summary.csv")
    write_csv(_safe_rows(summary.get("per_camera", [])), coverage_root / "per_camera_coverage.csv")
    write_csv(_safe_rows(summary.get("failed_cameras", [])), coverage_root / "failed_cameras.csv")
    write_csv(_safe_rows(summary.get("missing_predictions_after_generation", [])), coverage_root / "missing_predictions_after_generation.csv")
    write_json(summary.get("projection", {}), projection_root / "fullcam_projection_summary.json")
    failures = [row for row in summary.get("projection_rows", []) if not row.get("projection_valid")]
    write_csv(_safe_rows(failures), projection_root / "projection_failures.csv")
    write_json(summary.get("smoothness", {}), smoothness_root / "fullcam_smoothness_summary.json")
    write_csv(_safe_rows(summary.get("smoothness_rows", [])), smoothness_root / "fullcam_smoothness_per_camera.csv")
    write_csv(_safe_rows(summary.get("worst_remaining_jumps", [])), smoothness_root / "worst_remaining_jumps.csv")
    write_json(summary.get("source_metadata_completeness", {}), summaries_root / "source_metadata_completeness.json")
    write_csv(_dict_rows(summary.get("per_subset", {}), "subset"), summaries_root / "per_subset_summary.csv")
    write_csv(_dict_rows(summary.get("per_scene", {}), "scene_name"), summaries_root / "per_scene_summary.csv")
    write_csv(_safe_rows(summary.get("per_camera", [])), summaries_root / "per_camera_summary.csv")
    write_csv(_dict_rows(summary.get("per_class", {}), "class_id"), summaries_root / "per_class_summary.csv")
    eval_summary = {
        "status": "not_run",
        "reason": "Step 15G generation does not use GT. Run diagnostic eval separately for official_val/internal_holdout if needed.",
    }
    write_json(eval_summary, evaluation_root / "summary_eval.json")
    write_csv(_summary_rows(eval_summary), evaluation_root / "summary_eval.csv")


def _camera_row(item: FullCamItem, raw_stats: Dict[str, Any], stabilized_stats: Dict[str, Any]) -> Dict[str, Any]:
    expected = int(item.num_records or 0)
    raw_predictions = int(raw_stats.get("num_predictions", 0) or 0)
    stabilized_predictions = int(stabilized_stats.get("num_predictions", 0) or 0)
    return {
        "subset": item.subset,
        "split": item.split,
        "scene_name": item.scene_name,
        "camera_id": item.camera_id,
        "input_records_path": item.input_records_path,
        "raw_prediction_path": item.raw_prediction_path,
        "stabilized_prediction_path": item.stabilized_prediction_path,
        "input_exists": Path(item.input_records_path).exists(),
        "raw_exists": Path(item.raw_prediction_path).exists(),
        "stabilized_exists": Path(item.stabilized_prediction_path).exists(),
        "expected_records": expected,
        "raw_predictions": raw_predictions,
        "stabilized_predictions": stabilized_predictions,
        "raw_record_coverage": _rate(raw_predictions, expected),
        "stabilized_record_coverage": _rate(stabilized_predictions, expected),
        "raw_success": raw_stats.get("num_success", 0),
        "raw_failed": raw_stats.get("num_failed", 0),
        "raw_success_rate": raw_stats.get("success_rate"),
        "stabilized_success": stabilized_stats.get("num_success", 0),
        "stabilized_failed": stabilized_stats.get("num_failed", 0),
        "stabilized_success_rate": stabilized_stats.get("success_rate"),
        "raw_failure_reasons": raw_stats.get("failure_reasons", {}),
        "stabilized_failure_reasons": stabilized_stats.get("failure_reasons", {}),
        "source_metadata_completeness": stabilized_stats.get("source_metadata_completeness", {}),
        "per_class": stabilized_stats.get("per_class", {}),
    }


def _prediction_stats(path: Path) -> Dict[str, Any]:
    total = 0
    failed = 0
    failure_reasons = {}
    per_class = {}
    per_subset = {}
    metadata = {}
    for row in _read_prediction_rows(path):
        total += 1
        reason = row.get("failure_reason")
        if reason not in (None, ""):
            failed += 1
            _increment(failure_reasons, reason)
        _increment(per_class, row.get("class_id"))
        _increment(per_subset, row.get("subset"))
        for field in ["center_3d_source", "dimensions_3d_source", "yaw_source", "depth_source", "pseudo3d_method"]:
            if row.get(field) not in (None, "", "unknown"):
                key = "%s_complete" % field
                metadata[key] = metadata.get(key, 0) + 1
        if _bool(row.get("is_estimated_for_test")):
            metadata["is_estimated_for_test_set"] = metadata.get("is_estimated_for_test_set", 0) + 1
    metadata["total"] = total
    for key, value in list(metadata.items()):
        if key != "total":
            metadata["%s_rate" % key] = _rate(value, total)
    return {
        "num_predictions": total,
        "num_success": total - failed,
        "num_failed": failed,
        "success_rate": _rate(total - failed, total),
        "failure_reasons": failure_reasons,
        "per_class": per_class,
        "per_subset": per_subset,
        "source_metadata_completeness": metadata,
    }


def _projection_rows_for_item(item: FullCamItem, config: Dict[str, Any]) -> List[Dict[str, Any]]:
    calibration = load_scene_camera_calibration(_dataset_root(config), item.split, item.scene_name, item.camera_id)
    rows = []
    for row in _read_prediction_rows(Path(item.stabilized_prediction_path)):
        flat = _flatten_prediction(row)
        check = check_prediction_projection(flat, calibration)
        out = {
            "subset": item.subset,
            "scene_name": item.scene_name,
            "camera_id": item.camera_id,
            "frame_id": flat.get("frame_id"),
            "class_id": flat.get("class_id"),
            "local_track_id": flat.get("local_track_id"),
            "global_track_id": flat.get("global_track_id"),
        }
        out.update(check)
        rows.append(out)
    return rows


def _smoothness_for_item(item: FullCamItem, config: Dict[str, Any]) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    rows = [_flatten_prediction(row) for row in _read_prediction_rows(Path(item.stabilized_prediction_path))]
    smooth_cfg = config.get("smoothness", {})
    thresholds = {
        "suspicious_step_m": smooth_cfg.get("suspicious_step_m", 3.0),
        "invalid_step_m": smooth_cfg.get("invalid_step_m", 6.0),
        "suspicious_dimension_cv": smooth_cfg.get("suspicious_dimension_cv", 0.25),
        "invalid_dimension_cv": smooth_cfg.get("invalid_dimension_cv", 0.50),
        "yaw_jump_threshold": smooth_cfg.get("yaw_jump_threshold", 1.57),
    }
    summary = audit_pseudo3d_smoothness(rows, thresholds)
    status_distribution = summary.get("status_distribution", {})
    row = {
        "subset": item.subset,
        "scene_name": item.scene_name,
        "camera_id": item.camera_id,
        "object_count": summary.get("object_count"),
        "row_count": summary.get("row_count"),
        "good": status_distribution.get("good", 0),
        "suspicious": status_distribution.get("suspicious", 0),
        "invalid": status_distribution.get("invalid", 0),
        "step_p95": summary.get("step_distance_max_stats", {}).get("p95"),
        "step_p99": summary.get("step_distance_max_stats", {}).get("p99"),
        "step_max": summary.get("step_distance_max_stats", {}).get("max"),
    }
    jumps = worst_pseudo3d_jumps(rows, top_k=20)
    for jump in jumps:
        jump["subset"] = item.subset
        jump["scene_name"] = item.scene_name
        jump["camera_id"] = item.camera_id
    return row, jumps


def _summary_from_rows(
    items: List[FullCamItem],
    per_camera: List[Dict[str, Any]],
    projection_rows: List[Dict[str, Any]],
    smoothness_rows: List[Dict[str, Any]],
) -> Dict[str, Any]:
    total_expected = sum(int(row.get("expected_records", 0) or 0) for row in per_camera)
    total_raw = sum(int(row.get("raw_predictions", 0) or 0) for row in per_camera)
    total_stabilized = sum(int(row.get("stabilized_predictions", 0) or 0) for row in per_camera)
    raw_success = sum(int(row.get("raw_success", 0) or 0) for row in per_camera)
    stabilized_success = sum(int(row.get("stabilized_success", 0) or 0) for row in per_camera)
    failed_cameras = [row for row in per_camera if _camera_failed(row)]
    missing = [row for row in per_camera if not row.get("raw_exists") or not row.get("stabilized_exists")]
    return {
        "required_camera_files": len(items),
        "raw_files_existing": sum(1 for row in per_camera if row.get("raw_exists")),
        "stabilized_files_existing": sum(1 for row in per_camera if row.get("stabilized_exists")),
        "raw_files_missing": sum(1 for row in per_camera if not row.get("raw_exists")),
        "stabilized_files_missing": sum(1 for row in per_camera if not row.get("stabilized_exists")),
        "raw_file_coverage": _rate(sum(1 for row in per_camera if row.get("raw_exists")), len(items)),
        "stabilized_file_coverage": _rate(sum(1 for row in per_camera if row.get("stabilized_exists")), len(items)),
        "total_records_expected": total_expected,
        "total_raw_predictions": total_raw,
        "total_stabilized_predictions": total_stabilized,
        "raw_record_coverage": _rate(total_raw, total_expected),
        "stabilized_record_coverage": _rate(total_stabilized, total_expected),
        "success_rate_raw": _rate(raw_success, total_raw),
        "success_rate_stabilized": _rate(stabilized_success, total_stabilized),
        "failed_cameras": failed_cameras,
        "missing_predictions_after_generation": missing,
        "per_subset": _group_sum(per_camera, "subset", "stabilized_predictions"),
        "per_scene": _group_sum(per_camera, "scene_name", "stabilized_predictions"),
        "per_class": _merge_count_dicts([row.get("per_class", {}) for row in per_camera]),
        "failure_reasons": _merge_count_dicts([row.get("stabilized_failure_reasons", {}) for row in per_camera]),
        "source_metadata_completeness": _aggregate_metadata(per_camera, total_stabilized),
        "projection_rows_count": len(projection_rows),
        "smoothness_camera_count": len(smoothness_rows),
    }


def _aggregate_smoothness(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    object_count = sum(int(row.get("object_count", 0) or 0) for row in rows)
    row_count = sum(int(row.get("row_count", 0) or 0) for row in rows)
    return {
        "camera_count": len(rows),
        "object_count": object_count,
        "row_count": row_count,
        "status_distribution": {
            "good": sum(int(row.get("good", 0) or 0) for row in rows),
            "suspicious": sum(int(row.get("suspicious", 0) or 0) for row in rows),
            "invalid": sum(int(row.get("invalid", 0) or 0) for row in rows),
        },
        "step_p95_max_over_cameras": _max_present([row.get("step_p95") for row in rows]),
        "step_p99_max_over_cameras": _max_present([row.get("step_p99") for row in rows]),
        "step_max": _max_present([row.get("step_max") for row in rows]),
    }


def _compact_summary(summary: Dict[str, Any]) -> Dict[str, Any]:
    compact = dict(summary)
    compact.pop("per_camera", None)
    compact.pop("projection_rows", None)
    compact.pop("smoothness_rows", None)
    compact.pop("worst_remaining_jumps", None)
    return compact


def _read_prediction_rows(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def _flatten_prediction(row: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(row)
    center = row.get("center_3d")
    dims = row.get("dimensions_3d")
    bbox = row.get("bbox_xyxy")
    if isinstance(center, list) and len(center) >= 3:
        out["center_x"], out["center_y"], out["center_z"] = center[0], center[1], center[2]
    if isinstance(dims, list) and len(dims) >= 3:
        out["width_3d"], out["length_3d"], out["height_3d"] = dims[0], dims[1], dims[2]
    if isinstance(bbox, list) and len(bbox) >= 4:
        out["x1"], out["y1"], out["x2"], out["y2"] = bbox[0], bbox[1], bbox[2], bbox[3]
    return out


def _camera_failed(row: Dict[str, Any]) -> bool:
    if not row.get("input_exists") or not row.get("raw_exists") or not row.get("stabilized_exists"):
        return True
    if row.get("raw_predictions") != row.get("expected_records"):
        return True
    if row.get("stabilized_predictions") != row.get("expected_records"):
        return True
    return False


def _aggregate_metadata(rows: List[Dict[str, Any]], total_predictions: int) -> Dict[str, Any]:
    totals = {}
    for row in rows:
        metadata = row.get("source_metadata_completeness", {})
        if not isinstance(metadata, dict):
            continue
        for key, value in metadata.items():
            if key.endswith("_complete") or key == "is_estimated_for_test_set":
                totals[key] = totals.get(key, 0) + int(value or 0)
    for key, value in list(totals.items()):
        totals["%s_rate" % key] = _rate(value, total_predictions)
    totals["total"] = total_predictions
    return totals


def _group_sum(rows: List[Dict[str, Any]], key_field: str, value_field: str) -> Dict[str, int]:
    counts = {}
    for row in rows:
        key = str(row.get(key_field, ""))
        counts[key] = counts.get(key, 0) + int(row.get(value_field, 0) or 0)
    return counts


def _merge_count_dicts(items: List[Dict[str, Any]]) -> Dict[str, int]:
    counts = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        for key, value in item.items():
            counts[str(key)] = counts.get(str(key), 0) + int(value or 0)
    return counts


def _dict_rows(counts: Dict[str, Any], key_name: str) -> List[Dict[str, Any]]:
    return [{key_name: key, "count": value} for key, value in sorted(counts.items())]


def _summary_rows(summary: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows = []
    for key, value in sorted(summary.items()):
        if isinstance(value, (dict, list, tuple)):
            continue
        rows.append({"metric": key, "value": value})
    return rows


def _safe_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for row in rows:
        safe = {}
        for key, value in row.items():
            if isinstance(value, (dict, list, tuple)):
                safe[key] = json.dumps(value, sort_keys=True)
            else:
                safe[key] = value
        out.append(safe)
    return out


def _increment(counts: Dict[str, int], value: Any) -> None:
    key = str(value)
    counts[key] = counts.get(key, 0) + 1


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).lower() in ("true", "1", "yes")


def _rate(numerator: Any, denominator: Any) -> Optional[float]:
    try:
        den = float(denominator)
        if den == 0.0:
            return None
        return float(numerator) / den
    except (TypeError, ValueError):
        return None


def _max_present(values: List[Any]) -> Optional[float]:
    parsed = []
    for value in values:
        try:
            parsed.append(float(value))
        except (TypeError, ValueError):
            continue
    return max(parsed) if parsed else None


def _dataset_root(config: Dict[str, Any]) -> Path:
    return Path(config.get("paths", {}).get("dataset_root", "dataset/MTMC_Tracking_2026"))
