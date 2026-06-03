"""Writer scaffold for official Track 1 exports."""

import csv
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

from deep_oc_sort_3d.final_export.track1_export_types import Track1ExportSchema
from deep_oc_sort_3d.final_export.track1_mapping import GENERIC_TRACKING_COLUMNS, load_generic_tracking_csv


def export_track1_from_generic(
    generic_export_root: Union[str, Path],
    output_path: Union[str, Path],
    schema: Track1ExportSchema,
    mapping: Dict[str, str],
    subsets: Optional[List[str]] = None,
    scenes: Optional[List[str]] = None,
    show_progress: bool = True,
) -> Dict[str, Any]:
    """Export Track 1 output or an explicit unconfirmed preview from generic CSVs."""
    root = Path(generic_export_root)
    requested_output = Path(output_path)
    files = _find_generic_files(root, subsets=subsets, scenes=scenes)
    rows = _read_generic_rows(files, show_progress=show_progress)
    selected_subsets = sorted(set([item[0] for item in files]))
    selected_scenes = sorted(set([item[1] for item in files]))
    if not schema.schema_confirmed:
        preview_path = _preview_output_path(requested_output)
        rows_written = _write_unconfirmed_preview(rows, preview_path)
        return {
            "official_export_created": False,
            "reason": "schema_not_confirmed",
            "output_path": str(preview_path),
            "rows_written": rows_written,
            "scenes": selected_scenes,
            "subsets": selected_subsets,
            "schema_confirmed": False,
            "missing_columns": [],
            "warnings": [
                "Official Track 1 schema is not confirmed. Preview is not an official submission.",
            ],
        }
    missing_columns = [column for column in schema.columns if not mapping.get(column)]
    if missing_columns:
        raise ValueError("Cannot create official Track 1 export; missing columns: %s" % ",".join(missing_columns))
    output_rows = _collapse_rows_for_official_track1(rows, schema, mapping)
    rows_written = _write_confirmed_track1(output_rows, requested_output, schema, mapping, show_progress=show_progress)
    return {
        "official_export_created": True,
        "reason": "",
        "output_path": str(requested_output),
        "rows_written": rows_written,
        "input_rows": len(rows),
        "duplicate_rows_collapsed": max(0, len(rows) - rows_written),
        "scenes": selected_scenes,
        "subsets": selected_subsets,
        "schema_confirmed": True,
        "missing_columns": [],
        "warnings": [],
    }


def _find_generic_files(
    root: Path,
    subsets: Optional[List[str]],
    scenes: Optional[List[str]],
) -> List[Tuple[str, str, Path]]:
    subset_filter = None if subsets is None else set([str(item) for item in subsets])
    scene_filter = None if scenes is None else set([str(item) for item in scenes])
    output = []
    direct_files = sorted(root.glob("*.csv"))
    if direct_files:
        inferred_subset = root.name
        for path in direct_files:
            scene_name = path.stem
            if subset_filter is not None and inferred_subset not in subset_filter:
                continue
            if scene_filter is not None and scene_name not in scene_filter:
                continue
            output.append((inferred_subset, scene_name, path))
        return output
    for subset_dir in sorted(root.iterdir()) if root.exists() else []:
        if not subset_dir.is_dir():
            continue
        subset = subset_dir.name
        if subset_filter is not None and subset not in subset_filter:
            continue
        for path in sorted(subset_dir.glob("*.csv")):
            scene_name = path.stem
            if scene_filter is not None and scene_name not in scene_filter:
                continue
            output.append((subset, scene_name, path))
    return output


def _read_generic_rows(files: List[Tuple[str, str, Path]], show_progress: bool) -> List[Dict[str, Any]]:
    rows = []
    for subset, scene_name, path in _progress_iter(files, show_progress, "read generic exports", "file"):
        for row in load_generic_tracking_csv(path):
            row["_subset"] = subset
            row["_scene_file"] = scene_name
            rows.append(row)
    return sorted(
        rows,
        key=lambda item: (
            str(item.get("scene_name", "")),
            str(item.get("camera_id", "")),
            _safe_int(item.get("frame_id")),
            _safe_int(item.get("global_track_id")),
        ),
    )


def _preview_output_path(output_path: Path) -> Path:
    if output_path.name == "track1_unconfirmed_preview.csv":
        return output_path
    return output_path.parent / "track1_unconfirmed_preview.csv"


def _write_unconfirmed_preview(rows: List[Dict[str, Any]], output_path: Path) -> int:
    columns = [column for column in GENERIC_TRACKING_COLUMNS if _any_has_column(rows, column)]
    if not columns:
        columns = list(GENERIC_TRACKING_COLUMNS)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})
    return len(rows)


def _write_confirmed_track1(
    rows: List[Dict[str, Any]],
    output_path: Path,
    schema: Track1ExportSchema,
    mapping: Dict[str, str],
    show_progress: bool,
) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, delimiter=schema.delimiter)
        if schema.has_header is not False:
            writer.writerow(schema.columns)
        count = 0
        for row in _progress_iter(rows, show_progress, "write Track 1 rows", "row"):
            writer.writerow([_mapped_value(row, column, mapping[column], schema) for column in schema.columns])
            count += 1
    return count


def _collapse_rows_for_official_track1(
    rows: List[Dict[str, Any]],
    schema: Track1ExportSchema,
    mapping: Dict[str, str],
) -> List[Dict[str, Any]]:
    """Collapse camera-level duplicates for the no-camera Track 1 format.

    If multiple camera observations map to the same
    (scene_id, class_id, object_id, frame_id), keep the highest-confidence
    observation. This avoids duplicate object-frame records in the official
    3D output while leaving future fusion/ReID upgrades as TODO work.
    """
    required = ["scene_id", "class_id", "object_id", "frame_id"]
    if not all(column in schema.columns for column in required):
        return rows
    grouped = {}
    fallback_rows = []
    for row in rows:
        key = tuple([_mapped_value(row, column, mapping.get(column, ""), schema) for column in required])
        if any(value in (None, "") for value in key):
            fallback_rows.append(row)
            continue
        previous = grouped.get(key)
        if previous is None or _confidence(row) > _confidence(previous):
            grouped[key] = row
    collapsed = list(grouped.values()) + fallback_rows
    return sorted(
        collapsed,
        key=lambda item: (
            _safe_int(_mapped_value(item, "scene_id", mapping.get("scene_id", ""), schema)),
            _safe_int(_mapped_value(item, "class_id", mapping.get("class_id", ""), schema)),
            _safe_int(_mapped_value(item, "object_id", mapping.get("object_id", ""), schema)),
            _safe_int(_mapped_value(item, "frame_id", mapping.get("frame_id", ""), schema)),
        ),
    )


def _mapped_value(row: Dict[str, Any], official_column: str, generic_column: str, schema: Track1ExportSchema) -> Any:
    value = row.get(generic_column, "")
    if official_column == "scene_id":
        return _scene_name_to_id(value)
    if official_column == "object_id":
        return _safe_int(value)
    if schema.frame_indexing == "one_based" and official_column == "frame_id":
        return _safe_int(value) + 1
    if official_column in set(["class_id", "frame_id"]):
        return _safe_int(value)
    return value


def _any_has_column(rows: List[Dict[str, Any]], column: str) -> bool:
    return any(column in row for row in rows)


def _safe_int(value: Any) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return -1


def _scene_name_to_id(value: Any) -> Any:
    regex_result = re.search(r"(\d+)$", str(value))
    if regex_result is None:
        return ""
    return int(regex_result.group(1))


def _confidence(row: Dict[str, Any]) -> float:
    try:
        return float(row.get("confidence", 0.0))
    except (TypeError, ValueError):
        return 0.0


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
