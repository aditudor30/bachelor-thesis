"""Full-camera raw pseudo-3D generation orchestration."""

import csv
import json
import shutil
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from deep_oc_sort_3d.audit3d.audit3d_io import optional_int, progress_iter, write_csv, write_json
from deep_oc_sort_3d.pseudo3d.pseudo3d_config import load_pseudo3d_config
from deep_oc_sort_3d.pseudo3d.pseudo3d_estimator import Pseudo3DEstimator
from deep_oc_sort_3d.pseudo3d.pseudo3d_io import (
    load_scene_camera_calibration,
    prediction_summary,
    read_frame_record_inputs,
    write_pseudo3d_predictions_csv,
    write_pseudo3d_predictions_jsonl,
)
from deep_oc_sort_3d.pseudo3d.pseudo3d_priors import load_pseudo3d_priors
from deep_oc_sort_3d.pseudo3d.pseudo3d_types import Pseudo3DInput
from deep_oc_sort_3d.pseudo3d_fullcam.fullcam_discovery import FullCamItem, fullcam_item_to_dict


def generate_raw_for_item(item: FullCamItem, config: Dict[str, Any]) -> Dict[str, Any]:
    """Generate raw pseudo-3D predictions for one camera item."""
    estimator = _build_estimator(config)
    return _generate_raw_for_item(item, config, estimator, bool(config.get("step15g", {}).get("progress", True)))


def generate_raw_for_items(items: List[FullCamItem], config: Dict[str, Any], show_progress: bool = True) -> Dict[str, Any]:
    """Generate raw pseudo-3D predictions for many camera items."""
    estimator = _build_estimator(config)
    output_root = _output_root(config)
    report_root = output_root / "raw_generation_reports"
    summaries = []
    for item in progress_iter(items, show_progress, "generate fullcam pseudo3D raw", "camera"):
        row = _generate_raw_for_item(item, config, estimator, False)
        summaries.append(row)
        _write_raw_summaries(summaries, report_root)
    summary = _summarize_generation(summaries)
    write_json(summary, report_root / "summary_raw_generation.json")
    write_csv(_safe_rows(summaries), report_root / "summary_raw_generation.csv")
    return summary


def _generate_raw_for_item(item: FullCamItem, config: Dict[str, Any], estimator: Pseudo3DEstimator, show_progress: bool) -> Dict[str, Any]:
    output_jsonl = Path(item.raw_prediction_path)
    output_csv = output_jsonl.with_suffix(".csv")
    generation_cfg = config.get("generation", {})
    overwrite = bool(generation_cfg.get("overwrite", False))
    skip_existing = bool(generation_cfg.get("skip_existing", True))
    try:
        if output_jsonl.exists() and skip_existing and not overwrite:
            row = _prediction_file_summary(output_jsonl)
            row.update(_base_row(item, "skipped_existing", output_jsonl))
            _write_camera_report(config, item, row)
            return row
        legacy = _legacy_raw_path(config, item)
        if legacy.exists() and not overwrite:
            _copy_prediction_pair(legacy, output_jsonl)
            row = _prediction_file_summary(output_jsonl)
            row.update(_base_row(item, "reused_existing_raw", output_jsonl))
            _write_camera_report(config, item, row)
            return row
        if not Path(item.input_records_path).exists():
            row = _base_row(item, "error", output_jsonl)
            row.update({"error": "missing_input_records", "num_predictions": 0, "num_failed": 0, "success_rate": None})
            _write_camera_report(config, item, row)
            return row
        calibration = load_scene_camera_calibration(_dataset_root(config), item.split, item.scene_name, item.camera_id)
        inputs = _read_inputs(item, calibration, show_progress)
        outputs = [estimator.estimate(input_item) for input_item in inputs]
        write_pseudo3d_predictions_jsonl(outputs, output_jsonl)
        if bool(generation_cfg.get("output_format", {}).get("csv", True)):
            write_pseudo3d_predictions_csv(outputs, output_csv)
        row = prediction_summary(outputs)
        row.update(_base_row(item, "ok", output_jsonl))
        row["records_path"] = item.input_records_path
        _write_camera_report(config, item, row)
        return row
    except Exception as exc:
        row = _base_row(item, "error", output_jsonl)
        row.update({"error": str(exc), "traceback": traceback.format_exc(), "num_predictions": 0, "num_failed": 0, "success_rate": None})
        _write_camera_report(config, item, row)
        return row


def _build_estimator(config: Dict[str, Any]) -> Pseudo3DEstimator:
    generation_cfg = config.get("generation", {})
    paths = config.get("paths", {})
    config_path = generation_cfg.get("backend_config") or paths.get("pseudo3d_config") or "deep_oc_sort_3d/configs/pseudo3d_isolated_debug.yaml"
    priors_path = paths.get("class_priors", "")
    estimator_config = load_pseudo3d_config(config_path)
    priors = load_pseudo3d_priors(priors_path)
    return Pseudo3DEstimator(priors, estimator_config)


def _read_inputs(item: FullCamItem, calibration: Any, show_progress: bool) -> List[Pseudo3DInput]:
    path = Path(item.input_records_path)
    if path.suffix.lower() == ".jsonl":
        return _read_jsonl_inputs(path, item, calibration)
    return read_frame_record_inputs(path, item.subset, item.split, item.scene_name, item.camera_id, calibration, show_progress)


def _read_jsonl_inputs(path: Path, item: FullCamItem, calibration: Any) -> List[Pseudo3DInput]:
    rows = []
    width = _calib_int(calibration, "frame_width", "frameWidth") or 0
    height = _calib_int(calibration, "frame_height", "frameHeight") or 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            continue
        bbox = _bbox(row)
        rows.append(
            Pseudo3DInput(
                scene_name=str(row.get("scene_name") or item.scene_name),
                camera_id=str(row.get("camera_id") or item.camera_id),
                frame_id=int(float(row.get("frame_id", 0))),
                class_id=int(float(row.get("class_id", -1))),
                class_name=str(row.get("class_name", "")),
                bbox_xyxy=bbox,
                confidence=float(row.get("confidence", row.get("confidence_2d", 0.0)) or 0.0),
                image_width=width,
                image_height=height,
                calibration=calibration,
                track_id=optional_int(row.get("local_track_id")),
                subset=str(row.get("subset") or item.subset),
                split=str(row.get("split") or item.split),
                local_track_id=optional_int(row.get("local_track_id")),
                global_track_id=optional_int(row.get("global_track_id")),
                candidate_id=_optional_str(row.get("candidate_id")),
            )
        )
    return rows


def _bbox(row: Dict[str, Any]) -> Tuple[float, float, float, float]:
    value = row.get("bbox_xyxy")
    if isinstance(value, list) and len(value) >= 4:
        return (float(value[0]), float(value[1]), float(value[2]), float(value[3]))
    if isinstance(value, str):
        parts = [part.strip() for part in value.strip("[]()").replace(";", ",").split(",") if part.strip()]
        if len(parts) >= 4:
            return (float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]))
    return (
        float(row.get("x1", 0.0) or 0.0),
        float(row.get("y1", 0.0) or 0.0),
        float(row.get("x2", 0.0) or 0.0),
        float(row.get("y2", 0.0) or 0.0),
    )


def _prediction_file_summary(path: Path) -> Dict[str, Any]:
    total = 0
    failed = 0
    failure_reasons = {}
    per_class = {}
    per_subset = {}
    metadata_counts = {}
    for row in _iter_jsonl_dicts(path):
        total += 1
        reason = row.get("failure_reason")
        if reason not in (None, ""):
            failed += 1
            key = str(reason)
            failure_reasons[key] = failure_reasons.get(key, 0) + 1
        _increment(per_class, row.get("class_id"))
        _increment(per_subset, row.get("subset"))
        for field in ["center_3d_source", "dimensions_3d_source", "yaw_source", "depth_source", "pseudo3d_method"]:
            if row.get(field) not in (None, "", "unknown"):
                metadata_counts["%s_complete" % field] = metadata_counts.get("%s_complete" % field, 0) + 1
        if row.get("is_estimated_for_test") is True or str(row.get("is_estimated_for_test")).lower() in ("true", "1", "yes"):
            metadata_counts["is_estimated_for_test_set"] = metadata_counts.get("is_estimated_for_test_set", 0) + 1
    metadata_counts["total"] = total
    for key, value in list(metadata_counts.items()):
        if key != "total":
            metadata_counts["%s_rate" % key] = float(value) / float(total) if total else None
    return {
        "num_predictions": total,
        "num_success": total - failed,
        "num_failed": failed,
        "success_rate": float(total - failed) / float(total) if total else None,
        "failure_reasons": failure_reasons,
        "source_metadata_completeness": metadata_counts,
        "per_class": per_class,
        "per_subset": per_subset,
    }


def _iter_jsonl_dicts(path: Path) -> List[Dict[str, Any]]:
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


def _summarize_generation(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    ok_rows = [row for row in rows if row.get("status") in ("ok", "reused_existing_raw", "skipped_existing")]
    total = sum(int(row.get("num_predictions", 0) or 0) for row in ok_rows)
    failed = sum(int(row.get("num_failed", 0) or 0) for row in ok_rows)
    return {
        "camera_count": len(rows),
        "camera_ok": sum(1 for row in rows if row.get("status") == "ok"),
        "camera_reused_existing": sum(1 for row in rows if row.get("status") == "reused_existing_raw"),
        "camera_skipped": sum(1 for row in rows if row.get("status") == "skipped_existing"),
        "camera_errors": sum(1 for row in rows if row.get("status") == "error"),
        "num_predictions": total,
        "num_failed": failed,
        "success_rate": float(total - failed) / float(total) if total else None,
        "source_metadata_completeness": _aggregate_metadata(ok_rows, total),
        "per_class": _merge_count_dicts([row.get("per_class", {}) for row in ok_rows]),
        "per_subset": _merge_count_dicts([row.get("per_subset", {}) for row in ok_rows]),
        "failure_reasons": _merge_count_dicts([row.get("failure_reasons", {}) for row in ok_rows]),
    }


def _write_raw_summaries(rows: List[Dict[str, Any]], report_root: Path) -> None:
    summary = _summarize_generation(rows)
    write_json(summary, report_root / "summary_raw_generation.json")
    write_csv(_safe_rows(rows), report_root / "summary_raw_generation.csv")


def _write_camera_report(config: Dict[str, Any], item: FullCamItem, row: Dict[str, Any]) -> None:
    report_root = _output_root(config) / "raw_generation_reports" / item.subset / item.scene_name
    write_json(row, report_root / ("%s_raw_generation_report.json" % item.camera_id))
    write_csv(_safe_rows([row]), report_root / ("%s_raw_generation_report.csv" % item.camera_id))


def _base_row(item: FullCamItem, status: str, output_jsonl: Path) -> Dict[str, Any]:
    row = fullcam_item_to_dict(item)
    row.update({"status": status, "output_jsonl": str(output_jsonl), "output_csv": str(output_jsonl.with_suffix(".csv"))})
    return row


def _legacy_raw_path(config: Dict[str, Any], item: FullCamItem) -> Path:
    root = config.get("paths", {}).get("existing_raw_root")
    if not root:
        return Path("__missing__")
    return Path(root) / item.subset / item.scene_name / ("%s_pseudo3d_predictions.jsonl" % item.camera_id)


def _copy_prediction_pair(source_jsonl: Path, target_jsonl: Path) -> None:
    target_jsonl.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(str(source_jsonl), str(target_jsonl))
    source_csv = source_jsonl.with_suffix(".csv")
    if source_csv.exists():
        shutil.copy2(str(source_csv), str(target_jsonl.with_suffix(".csv")))


def _dataset_root(config: Dict[str, Any]) -> Path:
    return Path(config.get("paths", {}).get("dataset_root", "dataset/MTMC_Tracking_2026"))


def _output_root(config: Dict[str, Any]) -> Path:
    section = config.get("step15g", config.get("experiment", {}))
    return Path(section.get("output_root", "output/pseudo3d/baseline_v2_pseudo3d_fullcam"))


def _calib_int(calibration: Any, snake_key: str, json_key: str) -> Optional[int]:
    value = calibration.get(snake_key) if isinstance(calibration, dict) else getattr(calibration, snake_key, None)
    if value is None and isinstance(calibration, dict):
        value = calibration.get(json_key)
    return optional_int(value)


def _optional_str(value: Any) -> Optional[str]:
    if value in (None, ""):
        return None
    return str(value)


def _increment(counts: Dict[str, int], value: Any) -> None:
    key = str(value)
    counts[key] = counts.get(key, 0) + 1


def _merge_count_dicts(items: List[Dict[str, Any]]) -> Dict[str, int]:
    counts = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        for key, value in item.items():
            counts[str(key)] = counts.get(str(key), 0) + int(value or 0)
    return counts


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
        totals["%s_rate" % key] = float(value) / float(total_predictions) if total_predictions else None
    totals["total"] = total_predictions
    return totals


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
