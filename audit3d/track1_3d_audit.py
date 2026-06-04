"""Numeric audit for official Track 1 3D fields."""

from pathlib import Path
from typing import Any, Dict, List, Sequence, Union

from deep_oc_sort_3d.audit3d.audit3d_io import (
    finite_float,
    flatten_stats,
    group_rows,
    numeric_field_stats,
    optional_float,
    optional_int,
    progress_iter,
)


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

TRACK1_3D_FIELDS = ["x", "y", "z", "width", "length", "height", "yaw"]
TRACK1_DIMENSION_FIELDS = ["width", "length", "height"]


def read_track1_rows(path: Union[str, Path], show_progress: bool = True) -> List[Dict[str, Any]]:
    """Read official Track 1 rows.

    Expected format:
    scene_id class_id object_id frame_id x y z width length height yaw
    """
    track1_path = Path(path)
    rows = []
    if not track1_path.exists():
        return rows
    lines = track1_path.read_text(encoding="utf-8").splitlines()
    indexed_lines = list(enumerate(lines, start=1))
    for line_index, line in progress_iter(indexed_lines, show_progress, "read Track1 rows", "row"):
        stripped = line.strip()
        if not stripped:
            continue
        parts = stripped.split()
        row = {"row_index": line_index, "raw_column_count": len(parts)}
        for index, column in enumerate(TRACK1_COLUMNS):
            value = parts[index] if index < len(parts) else ""
            if column in ("scene_id", "class_id", "object_id", "frame_id"):
                row[column] = optional_int(value)
            else:
                row[column] = optional_float(value)
        rows.append(row)
    return rows


def compute_3d_field_stats(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute global numeric stats for Track 1 3D fields."""
    return {
        "row_count": len(rows),
        "field_stats": numeric_field_stats(rows, TRACK1_3D_FIELDS),
        "dimension_tuple_summary": _dimension_tuple_summary(rows, ["width", "length", "height"]),
        "invalid_row_count": sum(1 for row in rows if int(row.get("raw_column_count", 0)) != len(TRACK1_COLUMNS)),
    }


def compute_per_class_3d_stats(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Compute per-class Track 1 numeric summaries."""
    return _grouped_3d_stats(rows, ["class_id"])


def compute_per_scene_3d_stats(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Compute per-scene Track 1 numeric summaries."""
    return _grouped_3d_stats(rows, ["scene_id"])


def detect_extreme_3d_values(rows: List[Dict[str, Any]], config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Detect invalid, extreme, and likely default 3D values."""
    coordinate_abs_max = float(config.get("coordinate_abs_max", 10000.0))
    dimension_max = float(config.get("dimension_max", 20.0))
    yaw_min = config.get("yaw_min")
    yaw_max = config.get("yaw_max")
    rows_out = []
    for row in rows:
        row_flags = []
        for field in ["x", "y", "z"]:
            value = finite_float(row.get(field))
            if value is None:
                row_flags.append("%s_missing_or_nonfinite" % field)
            elif abs(value) > coordinate_abs_max:
                row_flags.append("%s_abs_gt_%s" % (field, coordinate_abs_max))
        for field in TRACK1_DIMENSION_FIELDS:
            value = finite_float(row.get(field))
            if value is None:
                row_flags.append("%s_missing_or_nonfinite" % field)
            elif value <= 0.0:
                row_flags.append("%s_non_positive" % field)
            elif value > dimension_max:
                row_flags.append("%s_gt_%s" % (field, dimension_max))
        yaw = finite_float(row.get("yaw"))
        if yaw is None:
            row_flags.append("yaw_missing_or_nonfinite")
        elif yaw_min is not None and yaw < float(yaw_min):
            row_flags.append("yaw_lt_%s" % yaw_min)
        elif yaw_max is not None and yaw > float(yaw_max):
            row_flags.append("yaw_gt_%s" % yaw_max)
        if int(row.get("raw_column_count", 0)) != len(TRACK1_COLUMNS):
            row_flags.append("invalid_column_count")
        if row_flags:
            rows_out.append(_extreme_row(row, ";".join(row_flags)))

    rows_out.extend(_detect_repeated_constant_dimensions(rows, config))
    return rows_out


def _grouped_3d_stats(rows: List[Dict[str, Any]], keys: Sequence[str]) -> List[Dict[str, Any]]:
    grouped = group_rows(rows, keys)
    summaries = []
    for key, group in sorted(grouped.items(), key=lambda item: tuple(str(value) for value in item[0])):
        summary = {"row_count": len(group)}
        for index, name in enumerate(keys):
            summary[name] = key[index]
        for field, stats in numeric_field_stats(group, TRACK1_3D_FIELDS).items():
            summary.update(flatten_stats(field, stats))
        summary.update(_dimension_tuple_summary(group, ["width", "length", "height"]))
        summaries.append(summary)
    return summaries


def _dimension_tuple_summary(rows: List[Dict[str, Any]], fields: Sequence[str]) -> Dict[str, Any]:
    tuples = []
    for row in rows:
        values = []
        valid = True
        for field in fields:
            value = finite_float(row.get(field))
            if value is None:
                valid = False
                break
            values.append(round(value, 6))
        if valid:
            tuples.append(tuple(values))
    if not tuples:
        return {
            "dimension_tuple_valid_count": 0,
            "dimension_tuple_unique_count": 0,
            "dimension_tuple_constant_ratio": None,
            "dimension_tuple_most_common": None,
        }
    counts = {}
    for value in tuples:
        counts[value] = counts.get(value, 0) + 1
    best = max(counts.items(), key=lambda item: item[1])
    return {
        "dimension_tuple_valid_count": len(tuples),
        "dimension_tuple_unique_count": len(counts),
        "dimension_tuple_constant_ratio": float(best[1]) / float(len(tuples)),
        "dimension_tuple_most_common": list(best[0]),
    }


def _detect_repeated_constant_dimensions(rows: List[Dict[str, Any]], config: Dict[str, Any]) -> List[Dict[str, Any]]:
    min_count = int(config.get("constant_dimension_min_count", 20))
    ratio_threshold = float(config.get("constant_dimension_ratio", 0.95))
    output = []
    grouped = group_rows(rows, ["scene_id", "class_id"])
    for key, group in sorted(grouped.items(), key=lambda item: tuple(str(value) for value in item[0])):
        summary = _dimension_tuple_summary(group, ["width", "length", "height"])
        valid_count = int(summary.get("dimension_tuple_valid_count") or 0)
        ratio = summary.get("dimension_tuple_constant_ratio")
        if valid_count >= min_count and ratio is not None and float(ratio) >= ratio_threshold:
            output.append(
                {
                    "row_index": "",
                    "scene_id": key[0],
                    "class_id": key[1],
                    "object_id": "",
                    "frame_id": "",
                    "issue": "repeated_constant_dimensions",
                    "value": summary.get("dimension_tuple_most_common"),
                    "count": valid_count,
                    "ratio": ratio,
                }
            )
    return output


def _extreme_row(row: Dict[str, Any], issue: str) -> Dict[str, Any]:
    output = {
        "row_index": row.get("row_index"),
        "scene_id": row.get("scene_id"),
        "class_id": row.get("class_id"),
        "object_id": row.get("object_id"),
        "frame_id": row.get("frame_id"),
        "issue": issue,
    }
    for field in TRACK1_3D_FIELDS:
        output[field] = row.get(field)
    return output


def stats_dict_to_rows(summary: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Convert the global field summary JSON to a flat CSV-friendly table."""
    rows = []
    field_stats = summary.get("field_stats", {})
    for field, stats in field_stats.items():
        row = {"field": field}
        if isinstance(stats, dict):
            row.update(stats)
        rows.append(row)
    tuple_summary = summary.get("dimension_tuple_summary", {})
    if tuple_summary:
        row = {"field": "dimension_tuple"}
        row.update(tuple_summary)
        rows.append(row)
    return rows

