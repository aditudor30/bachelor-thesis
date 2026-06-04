"""Full-camera pseudo-3D stabilization orchestration."""

import json
import shutil
import traceback
from pathlib import Path
from typing import Any, Dict, List

import yaml

from deep_oc_sort_3d.audit3d.audit3d_io import progress_iter, write_csv, write_json
from deep_oc_sort_3d.pseudo3d.pseudo3d_stabilization_io import (
    read_pseudo3d_outputs,
    write_smoothing_report_csv,
    write_smoothing_report_json,
    write_stabilized_outputs_csv,
    write_stabilized_outputs_jsonl,
)
from deep_oc_sort_3d.pseudo3d.pseudo3d_stabilizer import Pseudo3DStabilizer
from deep_oc_sort_3d.pseudo3d_fullcam.fullcam_discovery import FullCamItem, fullcam_item_to_dict


def stabilize_item(item: FullCamItem, config: Dict[str, Any]) -> Dict[str, Any]:
    """Stabilize one raw pseudo-3D camera file."""
    stabilizer_config = _load_stabilizer_config(config)
    return _stabilize_item(item, config, stabilizer_config)


def stabilize_items(items: List[FullCamItem], config: Dict[str, Any], show_progress: bool = True) -> Dict[str, Any]:
    """Stabilize raw pseudo-3D predictions for many camera files."""
    stabilizer_config = _load_stabilizer_config(config)
    report_root = _output_root(config) / "stabilization_reports"
    summaries = []
    for item in progress_iter(items, show_progress, "stabilize fullcam pseudo3D", "camera"):
        row = _stabilize_item(item, config, stabilizer_config)
        summaries.append(row)
        _write_stabilization_summaries(summaries, report_root)
    summary = _summarize_stabilization(summaries)
    write_json(summary, report_root / "summary_stabilization.json")
    write_csv(_safe_rows(summaries), report_root / "summary_stabilization.csv")
    return summary


def _stabilize_item(item: FullCamItem, config: Dict[str, Any], stabilizer_config: Dict[str, Any]) -> Dict[str, Any]:
    output_jsonl = Path(item.stabilized_prediction_path)
    output_csv = output_jsonl.with_suffix(".csv")
    stabilization_cfg = config.get("stabilization", {})
    overwrite = bool(stabilization_cfg.get("overwrite", False))
    skip_existing = bool(stabilization_cfg.get("skip_existing", True))
    try:
        if output_jsonl.exists() and skip_existing and not overwrite:
            row = _prediction_file_summary(output_jsonl)
            row.update(_base_row(item, "skipped_existing", output_jsonl))
            _write_camera_report(config, item, row, {})
            return row
        legacy = _legacy_stabilized_path(config, item)
        if legacy.exists() and not overwrite:
            _copy_prediction_pair(legacy, output_jsonl)
            row = _prediction_file_summary(output_jsonl)
            row.update(_base_row(item, "reused_existing_stabilized", output_jsonl))
            _write_camera_report(config, item, row, {})
            return row
        raw_path = Path(item.raw_prediction_path)
        if not raw_path.exists():
            row = _base_row(item, "error", output_jsonl)
            row.update({"error": "missing_raw_predictions", "num_predictions": 0, "num_failed": 0, "success_rate": None})
            _write_camera_report(config, item, row, {})
            return row
        outputs = read_pseudo3d_outputs(raw_path)
        stabilizer = Pseudo3DStabilizer(stabilizer_config)
        stabilized, report = stabilizer.stabilize_batch(outputs)
        write_stabilized_outputs_jsonl(stabilized, output_jsonl)
        if bool(stabilization_cfg.get("output_format", {}).get("csv", True)):
            write_stabilized_outputs_csv(stabilized, output_csv)
        row = _summary_row(report)
        row.update(_base_row(item, "ok", output_jsonl))
        _write_camera_report(config, item, row, report)
        return row
    except Exception as exc:
        row = _base_row(item, "error", output_jsonl)
        row.update({"error": str(exc), "traceback": traceback.format_exc(), "num_predictions": 0, "num_failed": 0, "success_rate": None})
        _write_camera_report(config, item, row, {})
        return row


def _load_stabilizer_config(config: Dict[str, Any]) -> Dict[str, Any]:
    path = config.get("stabilization", {}).get("config")
    if not path:
        return config.get("stabilization", {})
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def _summary_row(report: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "num_predictions": report.get("num_predictions"),
        "num_success": report.get("num_success"),
        "num_failed": report.get("num_failed"),
        "success_rate": report.get("success_rate"),
        "num_tracks": report.get("num_tracks"),
        "num_center_smoothed": report.get("num_center_smoothed"),
        "num_depth_smoothed": report.get("num_depth_smoothed"),
        "num_jump_corrected": report.get("num_jump_corrected"),
        "num_small_bbox_guarded": report.get("num_small_bbox_guarded"),
        "source_metadata_completeness": report.get("source_metadata_completeness", {}),
        "per_class": report.get("per_class", {}),
        "per_subset": report.get("per_subset", {}),
        "failure_reasons": report.get("failure_reasons", {}),
    }


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


def _summarize_stabilization(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    ok_rows = [row for row in rows if row.get("status") in ("ok", "reused_existing_stabilized", "skipped_existing")]
    total = sum(int(row.get("num_predictions", 0) or 0) for row in ok_rows)
    failed = sum(int(row.get("num_failed", 0) or 0) for row in ok_rows)
    return {
        "camera_count": len(rows),
        "camera_ok": sum(1 for row in rows if row.get("status") == "ok"),
        "camera_reused_existing": sum(1 for row in rows if row.get("status") == "reused_existing_stabilized"),
        "camera_skipped": sum(1 for row in rows if row.get("status") == "skipped_existing"),
        "camera_errors": sum(1 for row in rows if row.get("status") == "error"),
        "num_predictions": total,
        "num_failed": failed,
        "success_rate": float(total - failed) / float(total) if total else None,
        "num_tracks": sum(int(row.get("num_tracks", 0) or 0) for row in ok_rows),
        "num_center_smoothed": sum(int(row.get("num_center_smoothed", 0) or 0) for row in ok_rows),
        "num_depth_smoothed": sum(int(row.get("num_depth_smoothed", 0) or 0) for row in ok_rows),
        "num_jump_corrected": sum(int(row.get("num_jump_corrected", 0) or 0) for row in ok_rows),
        "num_small_bbox_guarded": sum(int(row.get("num_small_bbox_guarded", 0) or 0) for row in ok_rows),
        "source_metadata_completeness": _aggregate_metadata(ok_rows, total),
        "per_class": _merge_count_dicts([row.get("per_class", {}) for row in ok_rows]),
        "per_subset": _merge_count_dicts([row.get("per_subset", {}) for row in ok_rows]),
        "failure_reasons": _merge_count_dicts([row.get("failure_reasons", {}) for row in ok_rows]),
    }


def _write_stabilization_summaries(rows: List[Dict[str, Any]], report_root: Path) -> None:
    summary = _summarize_stabilization(rows)
    write_json(summary, report_root / "summary_stabilization.json")
    write_csv(_safe_rows(rows), report_root / "summary_stabilization.csv")


def _write_camera_report(config: Dict[str, Any], item: FullCamItem, row: Dict[str, Any], report: Dict[str, Any]) -> None:
    report_root = _output_root(config) / "stabilization_reports" / item.subset / item.scene_name
    write_json(row, report_root / ("%s_stabilization_summary.json" % item.camera_id))
    write_csv(_safe_rows([row]), report_root / ("%s_stabilization_summary.csv" % item.camera_id))
    if report:
        compact = dict(report)
        track_rows = list(compact.get("track_reports", []))
        compact["track_report_count"] = len(track_rows)
        write_smoothing_report_json(compact, report_root / ("%s_smoothing_report.json" % item.camera_id))
        write_smoothing_report_csv(track_rows, report_root / ("%s_smoothing_report.csv" % item.camera_id))


def _base_row(item: FullCamItem, status: str, output_jsonl: Path) -> Dict[str, Any]:
    row = fullcam_item_to_dict(item)
    row.update({"status": status, "output_jsonl": str(output_jsonl), "output_csv": str(output_jsonl.with_suffix(".csv"))})
    return row


def _legacy_stabilized_path(config: Dict[str, Any], item: FullCamItem) -> Path:
    root = config.get("paths", {}).get("existing_stabilized_root")
    if not root:
        return Path("__missing__")
    return Path(root) / item.subset / item.scene_name / ("%s_pseudo3d_stabilized.jsonl" % item.camera_id)


def _copy_prediction_pair(source_jsonl: Path, target_jsonl: Path) -> None:
    target_jsonl.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(str(source_jsonl), str(target_jsonl))
    source_csv = source_jsonl.with_suffix(".csv")
    if source_csv.exists():
        shutil.copy2(str(source_csv), str(target_jsonl.with_suffix(".csv")))


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


def _output_root(config: Dict[str, Any]) -> Path:
    section = config.get("step15g", config.get("experiment", {}))
    return Path(section.get("output_root", "output/pseudo3d/baseline_v2_pseudo3d_fullcam"))


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
