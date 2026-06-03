"""Validation scaffold for Track 1 exports."""

import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Union

from deep_oc_sort_3d.final_export.track1_export_types import Track1ExportSchema


def validate_track1_export(
    path: Union[str, Path],
    schema: Track1ExportSchema,
    show_progress: bool = True,
) -> Dict[str, Any]:
    """Validate a Track 1 output or unconfirmed preview."""
    output_path = Path(path)
    errors = []
    warnings = []
    if not output_path.exists():
        errors.append("missing_file")
        return _report(output_path, schema, errors, warnings, 0, official_validation=schema.schema_confirmed)
    if not schema.schema_confirmed:
        return {
            "path": str(output_path),
            "schema_confirmed": False,
            "official_validation": False,
            "reason": "schema_not_confirmed",
            "status": "ok",
            "num_errors": 0,
            "num_warnings": 1,
            "errors": [],
            "warnings": ["Preview exists, but this is not official Track 1 validation."],
            "rows": _count_rows(output_path),
        }
    rows = _read_rows(output_path, schema)
    if schema.has_header is True:
        header = rows["header"]
        if header != schema.columns:
            errors.append("header_mismatch")
    data_rows = rows["rows"]
    seen = set()
    indexed_rows = list(enumerate(data_rows, start=1))
    for row_index, row in _progress_iter(indexed_rows, show_progress, "validate Track 1 rows", "row"):
        if len(row) != len(schema.columns):
            errors.append("row_%d_column_count_%d_expected_%d" % (row_index, len(row), len(schema.columns)))
            continue
        named = {schema.columns[index]: row[index] for index in range(len(schema.columns))}
        _validate_named_row(named, row_index, schema, errors, warnings)
        duplicate_key = _duplicate_key(named)
        if duplicate_key is not None:
            if duplicate_key in seen:
                errors.append("duplicate_row_key:%s" % str(duplicate_key))
            seen.add(duplicate_key)
    return _report(output_path, schema, errors, warnings, len(data_rows), official_validation=True)


def write_track1_validation_report(report: Dict[str, Any], output_path: Union[str, Path]) -> None:
    """Write Track 1 validation report JSON."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")


def print_track1_validation_report(report: Dict[str, Any]) -> None:
    """Print a compact validation report."""
    print("schema_confirmed: %s" % report.get("schema_confirmed"))
    print("official_validation: %s" % report.get("official_validation"))
    print("status: %s" % report.get("status"))
    print("rows: %s" % report.get("rows"))
    print("errors: %s" % report.get("num_errors"))
    print("warnings: %s" % report.get("num_warnings"))
    for item in report.get("errors", []):
        print("  error: %s" % item)
    for item in report.get("warnings", []):
        print("  warning: %s" % item)


def _read_rows(path: Path, schema: Track1ExportSchema) -> Dict[str, Any]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle, delimiter=schema.delimiter)
        rows = list(reader)
    header = []
    data_rows = rows
    if schema.has_header is True and rows:
        header = rows[0]
        data_rows = rows[1:]
    elif schema.has_header is None and rows and rows[0] == schema.columns:
        header = rows[0]
        data_rows = rows[1:]
    return {"header": header, "rows": data_rows}


def _count_rows(path: Path) -> int:
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        rows = list(reader)
    if not rows:
        return 0
    return max(0, len(rows) - 1)


def _validate_named_row(
    row: Dict[str, Any],
    row_index: int,
    schema: Track1ExportSchema,
    errors: List[str],
    warnings: List[str],
) -> None:
    for column, value in row.items():
        lower = column.lower()
        if _looks_numeric_column(lower):
            number = _optional_float(value)
            if number is None:
                errors.append("row_%d_invalid_numeric:%s" % (row_index, column))
                continue
            if not math.isfinite(number):
                errors.append("row_%d_nan_or_inf:%s" % (row_index, column))
        if lower in ("frame", "frame_id"):
            frame = _optional_float(value)
            if frame is None:
                errors.append("row_%d_invalid_frame:%s" % (row_index, column))
            elif schema.frame_indexing == "one_based" and frame < 1:
                errors.append("row_%d_frame_not_one_based:%s" % (row_index, column))
            elif schema.frame_indexing == "zero_based" and frame < 0:
                errors.append("row_%d_frame_negative:%s" % (row_index, column))
        if lower in ("track_id", "global_track_id", "id"):
            track_id = _optional_float(value)
            if track_id is None or track_id < 0:
                errors.append("row_%d_invalid_track_id:%s" % (row_index, column))


def _looks_numeric_column(lower_name: str) -> bool:
    numeric_names = set(
        [
            "frame",
            "frame_id",
            "track_id",
            "global_track_id",
            "class_id",
            "confidence",
            "x",
            "y",
            "x1",
            "y1",
            "x2",
            "y2",
            "w",
            "h",
            "center_x",
            "center_y",
            "center_z",
            "width_3d",
            "length_3d",
            "height_3d",
            "yaw",
        ]
    )
    return lower_name in numeric_names


def _optional_float(value: Any) -> Any:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _duplicate_key(row: Dict[str, Any]) -> Any:
    keys = ["scene_name", "camera_id", "frame_id", "global_track_id"]
    if all(key in row for key in keys):
        return tuple(row[key] for key in keys)
    return None


def _report(
    path: Path,
    schema: Track1ExportSchema,
    errors: List[str],
    warnings: List[str],
    rows: int,
    official_validation: bool,
) -> Dict[str, Any]:
    return {
        "path": str(path),
        "schema_confirmed": schema.schema_confirmed,
        "official_validation": official_validation,
        "reason": "" if schema.schema_confirmed else "schema_not_confirmed",
        "status": "error" if errors else "ok",
        "num_errors": len(errors),
        "num_warnings": len(warnings),
        "errors": errors,
        "warnings": warnings,
        "rows": rows,
    }


def _progress_iter(values: List[Any], show_progress: bool, desc: str, unit: str) -> Iterable[Any]:
    if not show_progress:
        return values
    try:
        from tqdm import tqdm
    except ImportError:
        return _print_progress_iter(values, desc)
    return tqdm(values, desc=desc, unit=unit)


def _print_progress_iter(values: List[Any], desc: str) -> Iterable[Any]:
    total = len(values)
    for index, value in enumerate(values):
        if index == 0 or (index + 1) % 10000 == 0 or index + 1 == total:
            print("%s: %d/%d" % (desc, index + 1, total))
        yield value
