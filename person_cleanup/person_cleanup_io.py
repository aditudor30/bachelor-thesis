"""I/O helpers for Person cleanup experiments."""

import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import yaml


def load_yaml(path: Path) -> Dict[str, Any]:
    """Load a YAML dictionary."""
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def write_yaml(data: Dict[str, Any], path: Path) -> None:
    """Write YAML."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


def read_json(path: Path) -> Optional[Dict[str, Any]]:
    """Read a JSON dictionary if present."""
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def write_json(data: Dict[str, Any], path: Path) -> None:
    """Write JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def read_csv_rows(path: Path, add_subset_from_path: bool = False) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Read CSV rows and fieldnames."""
    if not path.exists():
        return [], []
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        rows = [dict(row) for row in reader]
    if add_subset_from_path:
        subset = infer_subset_from_path(path)
        for row in rows:
            row.setdefault("subset", subset)
    return rows, fieldnames


def write_csv_rows(rows: List[Dict[str, Any]], path: Path, fieldnames: Optional[List[str]] = None) -> None:
    """Write CSV rows."""
    if fieldnames is None:
        fieldnames = infer_fieldnames(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def infer_fieldnames(rows: List[Dict[str, Any]]) -> List[str]:
    """Infer CSV fieldnames preserving first-seen order."""
    fields = []
    for row in rows:
        for key in row.keys():
            if key not in fields and not str(key).startswith("_"):
                fields.append(str(key))
    return fields


def infer_subset_from_path(path: Path) -> str:
    """Infer subset from a generic or frame-record path."""
    parts = list(path.parts)
    for marker in ["generic_tracking_export", "frame_global_records"]:
        if marker in parts:
            index = parts.index(marker)
            if index + 1 < len(parts):
                return str(parts[index + 1])
    if len(parts) >= 2:
        return str(parts[-2])
    return "unknown"


def generic_csv_files(root: Path, subsets: Optional[List[str]] = None, scenes: Optional[List[str]] = None) -> List[Path]:
    """List generic scene CSVs."""
    subset_filter = None if subsets is None else set([str(item) for item in subsets])
    scene_filter = None if scenes is None else set([str(item) for item in scenes])
    files = []
    for path in sorted(root.rglob("*.csv")):
        subset = infer_subset_from_path(path)
        scene_name = path.stem
        if subset_filter is not None and subset not in subset_filter:
            continue
        if scene_filter is not None and scene_name not in scene_filter:
            continue
        files.append(path)
    return files


def frame_record_csv_files(root: Path, subsets: Optional[List[str]] = None, scenes: Optional[List[str]] = None) -> List[Path]:
    """List frame global record CSVs."""
    subset_filter = None if subsets is None else set([str(item) for item in subsets])
    scene_filter = None if scenes is None else set([str(item) for item in scenes])
    files = []
    for path in sorted(root.rglob("*_global_records.csv")):
        subset = infer_subset_from_path(path)
        scene_name = path.parent.name
        if subset_filter is not None and subset not in subset_filter:
            continue
        if scene_filter is not None and scene_name not in scene_filter:
            continue
        files.append(path)
    return files


def track_key(row: Dict[str, Any], subset: Optional[str] = None) -> Tuple[str, str, str, str]:
    """Return stable key: subset, scene, class_id, global_track_id."""
    row_subset = subset if subset is not None else str(row.get("subset", ""))
    return (
        str(row_subset),
        str(row.get("scene_name", "")),
        str(row.get("class_id", "")),
        str(row.get("global_track_id", "")),
    )


def progress_iter(values: Iterable[Any], show_progress: bool, desc: str, unit: str = "item") -> Iterable[Any]:
    """Iterate with tqdm if available, otherwise periodic prints."""
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
    """Return mean of numeric values."""
    parsed = [safe_float(value, None) for value in values]
    numeric = [value for value in parsed if value is not None]
    if not numeric:
        return None
    return float(sum(numeric)) / float(len(numeric))


def percentile(values: List[Any], p: float) -> Optional[float]:
    """Return a simple nearest-rank percentile."""
    parsed = [safe_float(value, None) for value in values]
    numeric = sorted([value for value in parsed if value is not None])
    if not numeric:
        return None
    if len(numeric) == 1:
        return float(numeric[0])
    index = int(round((float(p) / 100.0) * float(len(numeric) - 1)))
    index = max(0, min(len(numeric) - 1, index))
    return float(numeric[index])


def count_by(rows: List[Dict[str, Any]], field: str) -> Dict[str, int]:
    """Count rows by a field."""
    counts = {}
    for row in rows:
        key = str(row.get(field, ""))
        counts[key] = counts.get(key, 0) + 1
    return counts


def _print_progress_iter(values: Iterable[Any], desc: str) -> Iterable[Any]:
    total = len(values) if hasattr(values, "__len__") else None
    for index, value in enumerate(values):
        if total is None:
            if index == 0 or (index + 1) % 100 == 0:
                print("%s: item %d" % (desc, index + 1))
        elif index == 0 or (index + 1) % 10 == 0 or index + 1 == total:
            print("%s: %d/%d" % (desc, index + 1, total))
        yield value

