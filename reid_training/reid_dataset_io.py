"""I/O helpers for SmartSpaces ReID training dataset generation."""

import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

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


def write_jsonl(rows: Iterable[Dict[str, Any]], path: Path) -> int:
    """Write rows as JSONL and return count."""
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
            count += 1
    return count


def read_csv_rows(path: Path) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Read CSV rows and fieldnames."""
    if not path.exists():
        return [], []
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames or [])
        return [dict(row) for row in reader], fields


def write_csv_rows(rows: List[Dict[str, Any]], path: Path, fieldnames: Optional[List[str]] = None) -> None:
    """Write CSV rows."""
    if fieldnames is None:
        fieldnames = infer_fieldnames(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def infer_fieldnames(rows: List[Dict[str, Any]]) -> List[str]:
    """Infer field names preserving first-seen order."""
    fields: List[str] = []
    for row in rows:
        for key in row.keys():
            text = str(key)
            if text not in fields:
                fields.append(text)
    return fields


def write_text_lines(lines: List[str], path: Path) -> None:
    """Write text lines."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join([str(line) for line in lines]) + "\n", encoding="utf-8")


def progress_iter(values: Iterable[Any], show_progress: bool, desc: str, unit: str = "item") -> Iterable[Any]:
    """Iterate with tqdm if available, otherwise print periodic progress."""
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


def count_by(rows: List[Dict[str, Any]], field: str) -> Dict[str, int]:
    """Count rows by one field."""
    counts: Dict[str, int] = {}
    for row in rows:
        key = str(row.get(field, ""))
        counts[key] = counts.get(key, 0) + 1
    return counts


def group_by(rows: List[Dict[str, Any]], field: str) -> Dict[str, List[Dict[str, Any]]]:
    """Group rows by one field."""
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        key = str(row.get(field, ""))
        groups.setdefault(key, []).append(row)
    return groups


def numeric_summary(values: List[Any]) -> Dict[str, Any]:
    """Return simple numeric summary."""
    numeric = [safe_float(value, None) for value in values]
    numeric = sorted([value for value in numeric if value is not None])
    if not numeric:
        return {"count": 0, "min": None, "max": None, "mean": None, "median": None}
    return {
        "count": len(numeric),
        "min": float(numeric[0]),
        "max": float(numeric[-1]),
        "mean": float(sum(numeric)) / float(len(numeric)),
        "median": _percentile_sorted(numeric, 50),
        "p05": _percentile_sorted(numeric, 5),
        "p95": _percentile_sorted(numeric, 95),
    }


def _percentile_sorted(values: List[float], percentile: float) -> float:
    if len(values) == 1:
        return float(values[0])
    index = int(round(float(percentile) / 100.0 * float(len(values) - 1)))
    index = max(0, min(len(values) - 1, index))
    return float(values[index])


def _print_progress_iter(values: Iterable[Any], desc: str) -> Iterable[Any]:
    total = len(values) if hasattr(values, "__len__") else None
    for index, value in enumerate(values):
        if total is None:
            if index == 0 or (index + 1) % 1000 == 0:
                print("%s: item %d" % (desc, index + 1))
        elif index == 0 or (index + 1) % 100 == 0 or index + 1 == total:
            print("%s: %d/%d" % (desc, index + 1, total))
        yield value

