"""Shared utilities for Person ReID diagnostics."""

import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import yaml


def load_yaml(path: Path) -> Dict[str, Any]:
    """Load a YAML dictionary."""
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def write_json(data: Dict[str, Any], path: Path) -> None:
    """Write a JSON dictionary."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def read_json(path: Path) -> Optional[Dict[str, Any]]:
    """Read a JSON dictionary if present."""
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def read_csv_rows(path: Path) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Read CSV rows and fieldnames."""
    if not path.exists():
        return [], []
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames or [])
        return [dict(row) for row in reader], fields


def write_csv_rows(rows: List[Dict[str, Any]], path: Path, fieldnames: Optional[List[str]] = None) -> None:
    """Write rows to CSV."""
    if fieldnames is None:
        fieldnames = infer_fieldnames(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def infer_fieldnames(rows: List[Dict[str, Any]]) -> List[str]:
    """Infer CSV fieldnames preserving order."""
    fields = []
    for row in rows:
        for key in row.keys():
            if key not in fields:
                fields.append(str(key))
    return fields


def progress_iter(values: Iterable[Any], show_progress: bool, desc: str, unit: str = "item") -> Iterable[Any]:
    """Iterate with tqdm if available, otherwise print progress periodically."""
    if not show_progress:
        return values
    try:
        from tqdm import tqdm
    except ImportError:
        return _print_progress_iter(values, desc)
    return tqdm(values, desc=desc, unit=unit)


def safe_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    """Parse float safely."""
    if value in (None, ""):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def safe_int(value: Any, default: Optional[int] = None) -> Optional[int]:
    """Parse int safely."""
    number = safe_float(value, None)
    if number is None:
        return default
    return int(number)


def mean(values: List[Any]) -> Optional[float]:
    """Return mean for numeric values."""
    numeric = [safe_float(value, None) for value in values]
    numeric = [value for value in numeric if value is not None]
    if not numeric:
        return None
    return float(sum(numeric)) / float(len(numeric))


def percentile(values: List[Any], p: float) -> Optional[float]:
    """Return nearest-rank percentile."""
    numeric = [safe_float(value, None) for value in values]
    numeric = sorted([value for value in numeric if value is not None])
    if not numeric:
        return None
    if len(numeric) == 1:
        return float(numeric[0])
    index = int(round(float(p) / 100.0 * float(len(numeric) - 1)))
    index = max(0, min(len(numeric) - 1, index))
    return float(numeric[index])


def l2_normalize(vector: Any) -> np.ndarray:
    """L2-normalize a vector while preserving zero vectors."""
    arr = np.asarray(vector, dtype=float).reshape(-1)
    norm = float(np.linalg.norm(arr))
    if norm <= 1e-12:
        return arr.copy()
    return arr / norm


def cosine_similarity(left: Any, right: Any) -> float:
    """Compute cosine similarity."""
    a = np.asarray(left, dtype=float).reshape(-1)
    b = np.asarray(right, dtype=float).reshape(-1)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 1e-12:
        return 0.0
    return float(np.dot(a, b) / denom)


def count_by(rows: List[Dict[str, Any]], field: str) -> Dict[str, int]:
    """Count rows by field."""
    counts: Dict[str, int] = {}
    for row in rows:
        key = str(row.get(field, ""))
        counts[key] = counts.get(key, 0) + 1
    return counts


def _print_progress_iter(values: Iterable[Any], desc: str) -> Iterable[Any]:
    total = len(values) if hasattr(values, "__len__") else None
    for index, value in enumerate(values):
        if total is None:
            if index == 0 or (index + 1) % 1000 == 0:
                print("%s: item %d" % (desc, index + 1))
        elif index == 0 or (index + 1) % 100 == 0 or index + 1 == total:
            print("%s: %d/%d" % (desc, index + 1, total))
        yield value

