"""I/O helpers for visual review of fine-tuned Person ReID merges."""

import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import yaml

from deep_oc_sort_3d.person_association.person_association_io import parse_track_key, serialize_track_key
from deep_oc_sort_3d.person_cleanup.person_cleanup_io import (
    frame_record_csv_files,
    infer_fieldnames,
    load_yaml,
    mean,
    percentile,
    progress_iter,
    read_csv_rows,
    read_json,
    safe_float,
    safe_int,
    write_csv_rows,
    write_json,
    write_yaml,
)


TrackKey = Tuple[str, str, str, str]


def bool_from_any(value: Any) -> bool:
    """Parse common CSV boolean values."""
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in ("1", "true", "yes", "y")


def ensure_dirs(paths: Iterable[Path]) -> None:
    """Create all directories in `paths`."""
    for path in paths:
        Path(path).mkdir(parents=True, exist_ok=True)


def write_text(lines: Iterable[str], path: Path) -> None:
    """Write UTF-8 text lines."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join([str(line) for line in lines]) + "\n", encoding="utf-8")


def read_csv_dicts(path: Path) -> List[Dict[str, Any]]:
    """Read CSV rows, returning an empty list when missing."""
    rows, _fields = read_csv_rows(Path(path))
    return rows


def write_csv_dicts(rows: List[Dict[str, Any]], path: Path, fieldnames: Optional[List[str]] = None) -> None:
    """Write CSV rows with stable inferred field order."""
    write_csv_rows(rows, Path(path), fieldnames)


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Read a JSONL file defensively."""
    if not Path(path).exists():
        return []
    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if not text:
                continue
            try:
                value = json.loads(text)
            except ValueError:
                continue
            if isinstance(value, dict):
                rows.append(value)
    return rows


def parse_track_key_or_empty(value: Any) -> TrackKey:
    """Parse a track key, returning empty fields when malformed."""
    key = parse_track_key(value)
    if "" in key:
        return ("", "", "", "")
    return key


def track_key_text(value: Any) -> str:
    """Normalize a serialized track key."""
    key = parse_track_key_or_empty(value)
    if "" in key:
        return str(value or "")
    return serialize_track_key(key)


def stringify_path(path: Any) -> str:
    """Return a portable string path."""
    if path in (None, ""):
        return ""
    return str(Path(str(path)))


__all__ = [
    "Any",
    "Dict",
    "Iterable",
    "List",
    "Optional",
    "Path",
    "TrackKey",
    "Tuple",
    "bool_from_any",
    "csv",
    "ensure_dirs",
    "frame_record_csv_files",
    "infer_fieldnames",
    "load_yaml",
    "mean",
    "parse_track_key_or_empty",
    "percentile",
    "progress_iter",
    "read_csv_dicts",
    "read_csv_rows",
    "read_json",
    "read_jsonl",
    "safe_float",
    "safe_int",
    "stringify_path",
    "track_key_text",
    "write_csv_dicts",
    "write_csv_rows",
    "write_json",
    "write_text",
    "write_yaml",
    "yaml",
]

