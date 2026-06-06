"""I/O helpers for ReID ablation decision reports."""

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from deep_oc_sort_3d.person_association.person_association_io import (
    infer_fieldnames,
    load_yaml,
    progress_iter,
    read_csv_rows,
    read_json,
    safe_float,
    safe_int,
    write_csv_rows,
    write_json,
    write_yaml,
)


NOT_AVAILABLE = "not_available"


def ensure_dir(path: Path) -> None:
    """Create a directory if needed."""
    path.mkdir(parents=True, exist_ok=True)


def optional_number(value: Any) -> Optional[float]:
    """Return a numeric value or None."""
    return safe_float(value, None)


def metric_value(value: Any) -> Any:
    """Normalize missing values for JSON/CSV reporting."""
    if value in (None, ""):
        return NOT_AVAILABLE
    return value


def bool_from_errors(errors: Any) -> Any:
    """Return True when errors are zero, or not_available if unknown."""
    parsed = safe_int(errors, None)
    if parsed is None:
        return NOT_AVAILABLE
    return parsed == 0


__all__ = [
    "Any",
    "Dict",
    "Iterable",
    "List",
    "NOT_AVAILABLE",
    "Optional",
    "Path",
    "bool_from_errors",
    "ensure_dir",
    "infer_fieldnames",
    "load_yaml",
    "metric_value",
    "optional_number",
    "progress_iter",
    "read_csv_rows",
    "read_json",
    "safe_float",
    "safe_int",
    "write_csv_rows",
    "write_json",
    "write_yaml",
]

