"""I/O helpers for Person-aware association experiments."""

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from deep_oc_sort_3d.person_cleanup.person_cleanup_io import (
    count_by,
    frame_record_csv_files,
    generic_csv_files,
    infer_fieldnames,
    infer_subset_from_path,
    load_yaml,
    mean,
    percentile,
    progress_iter,
    read_csv_rows,
    read_json,
    safe_float,
    safe_int,
    track_key,
    write_csv_rows,
    write_json,
    write_yaml,
)


TrackKey = Tuple[str, str, str, str]


def parse_track_key(value: Any) -> TrackKey:
    """Parse a serialized track key."""
    parts = str(value).split("|")
    if len(parts) != 4:
        return ("", "", "", "")
    return (parts[0], parts[1], parts[2], parts[3])


def serialize_track_key(key: TrackKey) -> str:
    """Serialize a track key for CSV/JSON outputs."""
    return "%s|%s|%s|%s" % (key[0], key[1], key[2], key[3])


def row_track_key(row: Dict[str, Any], subset: Optional[str] = None) -> TrackKey:
    """Return the stable final-export track key."""
    return track_key(row, subset=subset)


def optional_list(value: Any) -> Optional[List[str]]:
    """Normalize optional config list values."""
    if value is None:
        return None
    if isinstance(value, str):
        return [value]
    return [str(item) for item in list(value)]


def ensure_dir(path: Path) -> None:
    """Create a directory if needed."""
    path.mkdir(parents=True, exist_ok=True)


def copy_filtered_csv_tree(
    source_root: Path,
    output_root: Path,
    files: List[Path],
    rows_by_path: Dict[str, List[Dict[str, Any]]],
    fieldnames_by_path: Dict[str, List[str]],
) -> Dict[str, Any]:
    """Write a mirrored CSV tree from in-memory rows."""
    rows_written = 0
    for path in files:
        key = str(path)
        relative = path.relative_to(source_root)
        rows = rows_by_path.get(key, [])
        fieldnames = fieldnames_by_path.get(key, infer_fieldnames(rows))
        write_csv_rows(rows, output_root / relative, fieldnames)
        rows_written += len(rows)
    return {"files": len(files), "rows_written": rows_written}


__all__ = [
    "Any",
    "Dict",
    "Iterable",
    "List",
    "Optional",
    "Path",
    "TrackKey",
    "Tuple",
    "copy_filtered_csv_tree",
    "count_by",
    "ensure_dir",
    "frame_record_csv_files",
    "generic_csv_files",
    "infer_fieldnames",
    "infer_subset_from_path",
    "load_yaml",
    "mean",
    "optional_list",
    "parse_track_key",
    "percentile",
    "progress_iter",
    "read_csv_rows",
    "read_json",
    "row_track_key",
    "safe_float",
    "safe_int",
    "serialize_track_key",
    "write_csv_rows",
    "write_json",
    "write_yaml",
]

