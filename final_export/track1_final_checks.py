"""Final sanity checks for official Track 1 submission files."""

import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union


TRACK1_COLUMNS = [
    "scene_id",
    "class_id",
    "object_id",
    "frame_id",
    "x",
    "y",
    "z",
    "width",
    "length",
    "height",
    "yaw",
]

INT_COLUMNS = set(["scene_id", "class_id", "object_id", "frame_id"])
FLOAT_COLUMNS = set(["x", "y", "z", "width", "length", "height", "yaw"])
DIMENSION_COLUMNS = set(["width", "length", "height"])
COORDINATE_COLUMNS = set(["x", "y", "z"])


def read_track1_txt(path: Union[str, Path]) -> List[Dict[str, Any]]:
    """Read Track 1 text file rows.

    Expected official format:
    scene_id class_id object_id frame_id x y z width length height yaw
    """
    track1_path = Path(path)
    rows = []
    if not track1_path.exists():
        return rows
    with track1_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            parts = stripped.split()
            row = {
                "_line_number": line_number,
                "_raw_line": stripped,
                "_num_columns": len(parts),
            }
            for index, column in enumerate(TRACK1_COLUMNS):
                row[column] = parts[index] if index < len(parts) else ""
            rows.append(row)
    return rows


def validate_track1_rows(
    rows: List[Dict[str, Any]],
    expected_scene_ids: Optional[List[int]] = None,
    valid_class_ids: Optional[List[int]] = None,
    show_progress: bool = True,
) -> Dict[str, Any]:
    """Validate final Track 1 rows."""
    expected_scene_set = None if expected_scene_ids is None else set([int(item) for item in expected_scene_ids])
    valid_class_set = None if valid_class_ids is None else set([int(item) for item in valid_class_ids])
    counts = {
        "empty_file": 1 if not rows else 0,
        "num_columns_invalid": 0,
        "non_numeric_values": 0,
        "nan_or_inf_values": 0,
        "negative_frame_id": 0,
        "invalid_scene_id": 0,
        "invalid_class_id": 0,
        "invalid_object_id": 0,
        "non_positive_dimensions": 0,
        "duplicate_key_count": 0,
        "sorting_issues": 0,
    }
    errors = []
    seen = set()
    sort_keys = []
    for row in _progress_iter(rows, show_progress, "validate track1 rows", "row"):
        line_number = int(row.get("_line_number", -1))
        if int(row.get("_num_columns", 0)) != len(TRACK1_COLUMNS):
            counts["num_columns_invalid"] += 1
            errors.append("line_%d_invalid_column_count" % line_number)
            continue
        parsed = _parse_row_numbers(row, line_number, counts, errors)
        if parsed is None:
            continue
        scene_id = int(parsed["scene_id"])
        class_id = int(parsed["class_id"])
        object_id = int(parsed["object_id"])
        frame_id = int(parsed["frame_id"])
        if expected_scene_set is not None and scene_id not in expected_scene_set:
            counts["invalid_scene_id"] += 1
            errors.append("line_%d_invalid_scene_id:%s" % (line_number, scene_id))
        if valid_class_set is not None and class_id not in valid_class_set:
            counts["invalid_class_id"] += 1
            errors.append("line_%d_invalid_class_id:%s" % (line_number, class_id))
        if object_id < 0:
            counts["invalid_object_id"] += 1
            errors.append("line_%d_invalid_object_id:%s" % (line_number, object_id))
        if frame_id < 0:
            counts["negative_frame_id"] += 1
            errors.append("line_%d_negative_frame_id:%s" % (line_number, frame_id))
        for column in DIMENSION_COLUMNS:
            if float(parsed[column]) <= 0.0:
                counts["non_positive_dimensions"] += 1
                errors.append("line_%d_non_positive_dimension:%s" % (line_number, column))
        key = (scene_id, class_id, object_id, frame_id)
        if key in seen:
            counts["duplicate_key_count"] += 1
            errors.append("line_%d_duplicate_key:%s" % (line_number, str(key)))
        seen.add(key)
        sort_keys.append(key)
    if sort_keys != sorted(sort_keys):
        counts["sorting_issues"] = 1
        errors.append("track1_rows_not_sorted_by_scene_class_object_frame")
    status = "error" if errors else "ok"
    distribution = compute_track1_distribution(rows)
    return {
        "status": status,
        "num_errors": len(errors),
        "num_warnings": 0,
        "errors": errors,
        "warnings": [],
        "checks": counts,
        "total_rows": len(rows),
        "distribution": distribution,
    }


def compute_track1_distribution(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute distribution and numeric ranges for Track 1 rows."""
    per_scene_rows = {}
    per_class_rows = {}
    per_scene_class_rows = {}
    objects_per_scene = {}
    objects_per_scene_class = {}
    frames_per_scene = {}
    values = {column: [] for column in list(COORDINATE_COLUMNS) + list(DIMENSION_COLUMNS) + ["yaw"]}
    for row in rows:
        parsed = _parse_row_numbers(row, int(row.get("_line_number", -1)), None, None)
        if parsed is None:
            continue
        scene_id = str(int(parsed["scene_id"]))
        class_id = str(int(parsed["class_id"]))
        object_id = int(parsed["object_id"])
        frame_id = int(parsed["frame_id"])
        per_scene_rows[scene_id] = per_scene_rows.get(scene_id, 0) + 1
        per_class_rows[class_id] = per_class_rows.get(class_id, 0) + 1
        scene_class_key = "%s/%s" % (scene_id, class_id)
        per_scene_class_rows[scene_class_key] = per_scene_class_rows.get(scene_class_key, 0) + 1
        objects_per_scene.setdefault(scene_id, set()).add(object_id)
        objects_per_scene_class.setdefault(scene_class_key, set()).add(object_id)
        frames_per_scene.setdefault(scene_id, []).append(frame_id)
        for column in values.keys():
            values[column].append(float(parsed[column]))
    return {
        "total_rows": len(rows),
        "per_scene_rows": per_scene_rows,
        "per_class_rows": per_class_rows,
        "per_scene_class_rows": per_scene_class_rows,
        "unique_objects_per_scene": _set_count_dict(objects_per_scene),
        "unique_objects_per_scene_class": _set_count_dict(objects_per_scene_class),
        "frame_min_max_per_scene": _min_max_dict(frames_per_scene),
        "coordinate_ranges": {column: _range(values[column]) for column in sorted(COORDINATE_COLUMNS)},
        "dimension_ranges": {column: _range(values[column]) for column in sorted(DIMENSION_COLUMNS)},
        "yaw_range": _range(values["yaw"]),
    }


def write_track1_final_report(report: Dict[str, Any], output_path: Union[str, Path]) -> None:
    """Write final Track 1 report JSON."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")


def print_track1_final_report(report: Dict[str, Any]) -> None:
    """Print a compact final Track 1 report."""
    print("status: %s" % report.get("status"))
    print("total_rows: %s" % report.get("total_rows"))
    print("num_errors: %s" % report.get("num_errors"))
    checks = report.get("checks", {})
    for key in sorted(checks.keys()):
        print("%s: %s" % (key, checks.get(key)))
    distribution = report.get("distribution", {})
    if distribution:
        print("per_scene_rows: %s" % json.dumps(distribution.get("per_scene_rows", {}), sort_keys=True))
        print("per_class_rows: %s" % json.dumps(distribution.get("per_class_rows", {}), sort_keys=True))


def _parse_row_numbers(
    row: Dict[str, Any],
    line_number: int,
    counts: Optional[Dict[str, int]],
    errors: Optional[List[str]],
) -> Optional[Dict[str, float]]:
    parsed = {}
    for column in TRACK1_COLUMNS:
        value = row.get(column, "")
        number = _parse_float(value)
        if number is None:
            if counts is not None:
                counts["non_numeric_values"] += 1
            if errors is not None:
                errors.append("line_%d_non_numeric:%s" % (line_number, column))
            return None
        if not math.isfinite(number):
            if counts is not None:
                counts["nan_or_inf_values"] += 1
            if errors is not None:
                errors.append("line_%d_nan_or_inf:%s" % (line_number, column))
            return None
        if column in INT_COLUMNS and abs(number - int(number)) > 1e-6:
            if counts is not None:
                counts["non_numeric_values"] += 1
            if errors is not None:
                errors.append("line_%d_non_integer:%s" % (line_number, column))
            return None
        parsed[column] = float(number)
    return parsed


def _parse_float(value: Any) -> Optional[float]:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _set_count_dict(values: Dict[str, Any]) -> Dict[str, int]:
    return {str(key): len(value) for key, value in values.items()}


def _min_max_dict(values: Dict[str, List[int]]) -> Dict[str, Dict[str, int]]:
    output = {}
    for key, items in values.items():
        output[str(key)] = {"min": min(items), "max": max(items)} if items else {"min": None, "max": None}
    return output


def _range(values: List[float]) -> Dict[str, Any]:
    if not values:
        return {"min": None, "max": None}
    return {"min": min(values), "max": max(values)}


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
