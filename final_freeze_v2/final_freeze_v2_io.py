"""Shared I/O helpers for final freeze v2 scripts."""

import csv
import json
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import yaml


NOT_AVAILABLE = "not_available"


def load_yaml(path: Path) -> Dict[str, Any]:
    """Load a YAML file as a dictionary."""
    if not Path(path).exists():
        return {}
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def write_yaml(data: Dict[str, Any], path: Path) -> None:
    """Write YAML data."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


def read_json(path: Path) -> Optional[Dict[str, Any]]:
    """Read a JSON dictionary if it exists."""
    if not Path(path).exists():
        return None
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def write_json(data: Dict[str, Any], path: Path) -> None:
    """Write JSON with indentation."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def read_csv_rows(path: Path) -> List[Dict[str, Any]]:
    """Read CSV rows, returning [] if missing."""
    if not Path(path).exists():
        return []
    with Path(path).open("r", newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


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
    """Infer CSV columns preserving first-seen order."""
    fields: List[str] = []
    for row in rows:
        for key in row.keys():
            text = str(key)
            if text not in fields:
                fields.append(text)
    return fields


def write_text(lines: Iterable[str], path: Path) -> None:
    """Write text lines."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join([str(line) for line in lines]) + "\n", encoding="utf-8")


def safe_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    """Parse a float safely."""
    if value in (None, "", NOT_AVAILABLE):
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


def metric_value(value: Any) -> Any:
    """Normalize missing metric values."""
    if value in (None, ""):
        return NOT_AVAILABLE
    return value


def count_text_rows(path: Path) -> Any:
    """Count non-empty lines in a text file."""
    if not Path(path).exists() or not Path(path).is_file():
        return NOT_AVAILABLE
    with Path(path).open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def copy_file_if_exists(source: Path, destination: Path) -> Optional[Path]:
    """Copy a file if present."""
    if not Path(source).exists() or not Path(source).is_file():
        return None
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(str(source), str(destination))
    return destination


def find_first_existing(paths: List[Path]) -> Optional[Path]:
    """Return first existing path in a list."""
    for path in paths:
        if Path(path).exists():
            return Path(path)
    return None


def find_first_track1(root: Path) -> Optional[Path]:
    """Find a Track1 text file under a root."""
    root = Path(root)
    candidates = [
        root / "track1.txt",
        root / "track1_submission" / "track1.txt",
        root / "submission" / "track1.txt",
    ]
    found = find_first_existing(candidates)
    if found is not None and found.is_file():
        return found
    if root.exists() and root.is_dir():
        files = sorted(root.rglob("track1.txt"))
        if files:
            return files[0]
    return None


def progress_iter(values: Iterable[Any], show_progress: bool, desc: str, unit: str = "item") -> Iterable[Any]:
    """Iterate with tqdm if available, otherwise print periodic progress."""
    if not show_progress:
        return values
    try:
        from tqdm import tqdm
    except ImportError:
        return _print_progress_iter(values, desc)
    return tqdm(values, desc=desc, unit=unit)


def _print_progress_iter(values: Iterable[Any], desc: str) -> Iterable[Any]:
    total = len(values) if hasattr(values, "__len__") else None
    for index, value in enumerate(values):
        if total is None:
            if index == 0 or (index + 1) % 100 == 0:
                print("%s: item %d" % (desc, index + 1))
        elif index == 0 or (index + 1) % 10 == 0 or index + 1 == total:
            print("%s: %d/%d" % (desc, index + 1, total))
        yield value

