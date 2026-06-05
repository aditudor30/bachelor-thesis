"""Small I/O helpers for global tuning runs."""

import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import yaml


def load_yaml(path: Path) -> Dict[str, Any]:
    """Load a YAML file as a dictionary."""
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def write_yaml(data: Dict[str, Any], path: Path) -> None:
    """Write a dictionary as YAML."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


def read_json(path: Path) -> Optional[Dict[str, Any]]:
    """Read JSON dictionary if present."""
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def write_json(data: Dict[str, Any], path: Path) -> None:
    """Write JSON dictionary."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    """Read CSV rows; return an empty list if the file is missing."""
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def write_csv_rows(rows: List[Dict[str, Any]], path: Path, fieldnames: Optional[List[str]] = None) -> None:
    """Write rows as CSV."""
    if fieldnames is None:
        fieldnames = _infer_fieldnames(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def count_csv_rows(path: Path) -> int:
    """Count data rows in a CSV file without loading it all."""
    if not path.exists():
        return 0
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        count = sum(1 for _row in reader)
    return max(0, int(count) - 1)


def progress_iter(values: Iterable[Any], show_progress: bool, desc: str, unit: str = "item") -> Iterable[Any]:
    """Iterate with tqdm when available, otherwise periodic prints."""
    if not show_progress:
        return values
    try:
        from tqdm import tqdm
    except ImportError:
        return _print_progress_iter(values, desc)
    return tqdm(values, desc=desc, unit=unit)


def safe_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    """Parse a float safely."""
    if value in (None, ""):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def safe_int(value: Any, default: Optional[int] = None) -> Optional[int]:
    """Parse an int safely."""
    number = safe_float(value, None)
    if number is None:
        return default
    return int(number)


def ratio(numerator: Any, denominator: Any) -> Optional[float]:
    """Return numerator / denominator or None."""
    num = safe_float(numerator, None)
    den = safe_float(denominator, None)
    if num is None or den is None or den == 0:
        return None
    return float(num) / float(den)


def mean(values: List[Any]) -> Optional[float]:
    """Return arithmetic mean for numeric values."""
    parsed = [safe_float(value, None) for value in values]
    numeric = [value for value in parsed if value is not None]
    if not numeric:
        return None
    return float(sum(numeric)) / float(len(numeric))


def _infer_fieldnames(rows: List[Dict[str, Any]]) -> List[str]:
    fields = []
    for row in rows:
        for key in row.keys():
            if key not in fields:
                fields.append(str(key))
    return fields


def _print_progress_iter(values: Iterable[Any], desc: str) -> Iterable[Any]:
    total = len(values) if hasattr(values, "__len__") else None
    for index, value in enumerate(values):
        if total is None:
            if index == 0 or (index + 1) % 100 == 0:
                print("%s: item %d" % (desc, index + 1))
        elif index == 0 or (index + 1) % 10 == 0 or index + 1 == total:
            print("%s: %d/%d" % (desc, index + 1, total))
        yield value

