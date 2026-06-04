"""Audit 3D fields in generic frame-level exports."""

from pathlib import Path
from typing import Any, Dict, List, Sequence, Set, Tuple, Union

from deep_oc_sort_3d.audit3d.audit3d_io import (
    finite_float,
    flatten_stats,
    group_rows,
    iter_data_files,
    numeric_field_stats,
    optional_float,
    optional_int,
    progress_iter,
    read_csv_dicts,
    scene_id_from_name,
)


GENERIC_EXPORT_FIELDS = [
    "scene_name",
    "camera_id",
    "frame_id",
    "global_track_id",
    "class_id",
    "class_name",
    "confidence",
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

GENERIC_3D_FIELDS = ["center_x", "center_y", "center_z", "width_3d", "length_3d", "height_3d", "yaw"]
GENERIC_NUMERIC_FIELDS = [
    "frame_id",
    "global_track_id",
    "class_id",
    "confidence",
    "x1",
    "y1",
    "x2",
    "y2",
    "w",
    "h",
] + GENERIC_3D_FIELDS


def read_generic_export_rows(root_or_file: Union[str, Path], show_progress: bool = True) -> List[Dict[str, Any]]:
    """Read generic export CSV rows from a file or directory tree."""
    rows = []
    for path in progress_iter(iter_data_files(root_or_file, [".csv"]), show_progress, "read generic export files", "file"):
        file_rows = read_csv_dicts(path)
        if not file_rows:
            continue
        if not _looks_like_generic_export(file_rows[0]):
            continue
        for row in progress_iter(file_rows, show_progress, "read generic rows %s" % path.name, "row"):
            parsed = dict(row)
            parsed["source_file"] = str(path)
            parsed["subset"] = _infer_subset(path)
            parsed["scene_id"] = scene_id_from_name(parsed.get("scene_name"))
            for field in GENERIC_NUMERIC_FIELDS:
                if field in parsed:
                    if field in ("frame_id", "global_track_id", "class_id"):
                        parsed[field] = optional_int(parsed.get(field))
                    else:
                        parsed[field] = optional_float(parsed.get(field))
            rows.append(parsed)
    return rows


def compute_generic_3d_stats(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute global 3D field stats for generic exports."""
    return {
        "row_count": len(rows),
        "field_stats": numeric_field_stats(rows, GENERIC_3D_FIELDS),
        "subsets": _distribution(rows, "subset"),
        "source_files": len(set([str(row.get("source_file", "")) for row in rows])),
    }


def compute_generic_per_class_stats(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Compute per-class generic export 3D summaries."""
    return _grouped_3d_stats(rows, ["class_id", "class_name"])


def compute_generic_per_scene_stats(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Compute per-scene generic export 3D summaries."""
    return _grouped_3d_stats(rows, ["scene_name", "scene_id"])


def compare_generic_vs_track1(generic_rows: List[Dict[str, Any]], track1_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compare generic export 3D ranges and distributions with Track 1."""
    generic_test_rows = _generic_rows_that_match_track1_scenes(generic_rows, track1_rows)
    generic_key_count = len(_generic_dedup_keys(generic_test_rows))
    track1_key_count = len(_track1_keys(track1_rows))
    return {
        "generic_rows_total": len(generic_rows),
        "generic_rows_matching_track1_scenes": len(generic_test_rows),
        "track1_rows_total": len(track1_rows),
        "generic_unique_track1_keys": generic_key_count,
        "track1_unique_keys": track1_key_count,
        "dedup_difference_rows": len(generic_test_rows) - len(track1_rows),
        "dedup_difference_unique_keys": generic_key_count - track1_key_count,
        "generic_class_distribution": _distribution(generic_test_rows, "class_id"),
        "track1_class_distribution": _distribution(track1_rows, "class_id"),
        "generic_scene_distribution": _distribution(generic_test_rows, "scene_id"),
        "track1_scene_distribution": _distribution(track1_rows, "scene_id"),
        "generic_coordinate_ranges": _range_summary(generic_test_rows, ["center_x", "center_y", "center_z"]),
        "track1_coordinate_ranges": _range_summary(track1_rows, ["x", "y", "z"]),
        "generic_dimension_ranges": _range_summary(generic_test_rows, ["width_3d", "length_3d", "height_3d"]),
        "track1_dimension_ranges": _range_summary(track1_rows, ["width", "length", "height"]),
    }


def stats_dict_to_rows(summary: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Convert generic field summary to rows."""
    rows = []
    for field, stats in summary.get("field_stats", {}).items():
        row = {"field": field}
        if isinstance(stats, dict):
            row.update(stats)
        rows.append(row)
    return rows


def _looks_like_generic_export(row: Dict[str, Any]) -> bool:
    return all(field in row for field in ["scene_name", "camera_id", "frame_id", "global_track_id", "center_x"])


def _grouped_3d_stats(rows: List[Dict[str, Any]], keys: Sequence[str]) -> List[Dict[str, Any]]:
    grouped = group_rows(rows, keys)
    summaries = []
    for key, group in sorted(grouped.items(), key=lambda item: tuple(str(value) for value in item[0])):
        summary = {"row_count": len(group)}
        for index, name in enumerate(keys):
            summary[name] = key[index]
        for field, stats in numeric_field_stats(group, GENERIC_3D_FIELDS).items():
            summary.update(flatten_stats(field, stats))
        summaries.append(summary)
    return summaries


def _infer_subset(path: Path) -> str:
    parts = [part.lower() for part in path.parts]
    for subset in ["official_val", "internal_holdout", "test", "train", "val"]:
        if subset in parts:
            return subset
    return ""


def _distribution(rows: List[Dict[str, Any]], key: str) -> Dict[str, int]:
    counts = {}
    for row in rows:
        value = str(row.get(key, ""))
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items(), key=lambda item: item[0]))


def _range_summary(rows: List[Dict[str, Any]], fields: Sequence[str]) -> Dict[str, Dict[str, Any]]:
    summary = {}
    for field in fields:
        values = [finite_float(row.get(field)) for row in rows]
        finite = [value for value in values if value is not None]
        summary[field] = {
            "min": min(finite) if finite else None,
            "max": max(finite) if finite else None,
            "valid_count": len(finite),
        }
    return summary


def _generic_rows_that_match_track1_scenes(
    generic_rows: List[Dict[str, Any]],
    track1_rows: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    scene_ids = set([row.get("scene_id") for row in track1_rows if row.get("scene_id") is not None])
    if not scene_ids:
        return generic_rows
    return [row for row in generic_rows if row.get("scene_id") in scene_ids]


def _generic_dedup_keys(rows: List[Dict[str, Any]]) -> Set[Tuple[Any, Any, Any, Any]]:
    keys = set()
    for row in rows:
        keys.add((row.get("scene_id"), row.get("class_id"), row.get("global_track_id"), row.get("frame_id")))
    return keys


def _track1_keys(rows: List[Dict[str, Any]]) -> Set[Tuple[Any, Any, Any, Any]]:
    keys = set()
    for row in rows:
        keys.add((row.get("scene_id"), row.get("class_id"), row.get("object_id"), row.get("frame_id")))
    return keys
