"""Validation utilities for final MVP export outputs."""

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from deep_oc_sort_3d.final_export.generic_export import GENERIC_EXPORT_FIELDS, read_global_frame_records_file
from deep_oc_sort_3d.final_export.global_frame_types import GlobalFrameRecord


def validate_global_frame_records(records: List[GlobalFrameRecord]) -> Dict[str, Any]:
    """Validate frame-level global records."""
    errors = []
    warnings = []
    info = []
    if not records:
        warnings.append("empty_records")
    seen = set()
    class_by_global = {}
    scene_by_global = {}
    sorted_keys = []
    for record in records:
        key = (record.scene_name, record.camera_id, record.frame_id, record.global_track_id, record.detection_id)
        sorted_keys.append(key)
        if key in seen:
            errors.append("duplicate_row:%s" % str(key))
        seen.add(key)
        _validate_one_record(record, errors, warnings)
        if record.global_track_id is not None:
            class_by_global.setdefault(record.global_track_id, set()).add(record.class_id)
            scene_by_global.setdefault(record.global_track_id, set()).add(record.scene_name)
    for global_track_id, class_ids in class_by_global.items():
        if len(class_ids) > 1:
            errors.append("global_track_class_inconsistency:%s" % global_track_id)
    for global_track_id, scene_names in scene_by_global.items():
        if len(scene_names) > 1:
            errors.append("global_track_scene_inconsistency:%s" % global_track_id)
    if sorted_keys != sorted(sorted_keys):
        warnings.append("records_not_sorted")
    info.append("records=%d" % len(records))
    return _report(errors, warnings, info)


def validate_generic_tracking_export(path: Path) -> Dict[str, Any]:
    """Validate generic per-scene tracking CSV."""
    errors = []
    warnings = []
    info = []
    if not path.exists():
        return _report(["missing_file"], warnings, info)
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            return _report([], ["empty_file"], info)
        missing = [field for field in GENERIC_EXPORT_FIELDS if field not in reader.fieldnames]
        if missing:
            errors.append("missing_columns:%s" % ",".join(missing))
        seen = set()
        class_by_global = {}
        scene_by_global = {}
        rows = list(reader)
    if not rows:
        warnings.append("empty_export")
    for row in rows:
        key = (row.get("scene_name"), row.get("camera_id"), row.get("frame_id"), row.get("global_track_id"))
        if key in seen:
            errors.append("duplicate_row:%s" % str(key))
        seen.add(key)
        global_track_id = _optional_int(row.get("global_track_id"))
        if global_track_id is None:
            errors.append("missing_global_track_id")
        class_id = _optional_int(row.get("class_id"))
        if class_id is None or class_id < 0:
            errors.append("invalid_class_id")
        if global_track_id is not None and class_id is not None:
            class_by_global.setdefault(global_track_id, set()).add(class_id)
            scene_by_global.setdefault(global_track_id, set()).add(str(row.get("scene_name", "")))
        _validate_bbox_values(row, errors)
        _validate_float_fields(row, errors)
    for global_track_id, class_ids in class_by_global.items():
        if len(class_ids) > 1:
            errors.append("global_track_class_inconsistency:%s" % global_track_id)
    for global_track_id, scene_names in scene_by_global.items():
        if len(scene_names) > 1:
            errors.append("global_track_scene_inconsistency:%s" % global_track_id)
    info.append("rows=%d" % len(rows))
    return _report(errors, warnings, info)


def write_validation_report(report: Dict[str, Any], path: Path) -> None:
    """Write validation report as JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")


def print_validation_report(report: Dict[str, Any]) -> None:
    """Print compact validation report."""
    print("errors: %s" % report.get("num_errors"))
    print("warnings: %s" % report.get("num_warnings"))
    print("info: %s" % json.dumps(report.get("info", [])))


def validate_global_frame_record_file(path: Path) -> Dict[str, Any]:
    """Validate one frame record CSV/JSONL file."""
    return validate_global_frame_records(read_global_frame_records_file(path))


def _validate_one_record(record: GlobalFrameRecord, errors: List[str], warnings: List[str]) -> None:
    if record.global_track_id is None:
        warnings.append("missing_global_track_id")
    if record.frame_id < 0:
        errors.append("negative_frame_id")
    if record.class_id < 0:
        errors.append("invalid_class_id")
    if record.confidence < 0.0 or record.confidence > 1.0:
        warnings.append("confidence_out_of_range")
    x1, y1, x2, y2 = record.bbox_xyxy
    if x2 <= x1 or y2 <= y1:
        errors.append("invalid_bbox")
    if record.center_3d is not None:
        arr = np.asarray(record.center_3d, dtype=float)
        if not np.all(np.isfinite(arr)):
            errors.append("center_3d_nan_or_inf")


def _validate_bbox_values(row: Dict[str, Any], errors: List[str]) -> None:
    x1 = _optional_float(row.get("x1"))
    y1 = _optional_float(row.get("y1"))
    x2 = _optional_float(row.get("x2"))
    y2 = _optional_float(row.get("y2"))
    if x1 is None or y1 is None or x2 is None or y2 is None:
        errors.append("missing_bbox")
        return
    if x2 <= x1 or y2 <= y1:
        errors.append("invalid_bbox")


def _validate_float_fields(row: Dict[str, Any], errors: List[str]) -> None:
    for key in ["center_x", "center_y", "center_z", "width_3d", "length_3d", "height_3d", "yaw"]:
        value = row.get(key)
        if value in (None, ""):
            continue
        number = float(value)
        if not np.isfinite(number):
            errors.append("nan_or_inf:%s" % key)


def _report(errors: List[str], warnings: List[str], info: List[str]) -> Dict[str, Any]:
    return {
        "errors": errors,
        "warnings": warnings,
        "info": info,
        "num_errors": len(errors),
        "num_warnings": len(warnings),
        "status": "error" if errors else "ok",
    }


def _optional_float(value: Any) -> Any:
    if value in (None, ""):
        return None
    return float(value)


def _optional_int(value: Any) -> Any:
    if value in (None, ""):
        return None
    return int(float(value))
