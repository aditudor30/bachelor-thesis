"""I/O and streaming helpers for fragmentation audit scripts."""

import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional

import yaml


SKIP_DIR_NAMES = set(["summaries", "summary", "report", "reports", "plots_optional", "diagnostics", "visualizations", "eval", "figures"])


def load_yaml(path: Path) -> Dict[str, Any]:
    """Load a YAML file, returning an empty dict for empty documents."""
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def read_json(path: Path) -> Dict[str, Any]:
    """Read a JSON file if it exists."""
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(data: Dict[str, Any], path: Path) -> None:
    """Write pretty JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def write_csv(rows: List[Dict[str, Any]], path: Path, fieldnames: Optional[List[str]] = None) -> None:
    """Write a list of dictionaries to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = fieldnames if fieldnames is not None else sorted(set([key for row in rows for key in row.keys()]))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def iter_csv_rows(path: Path) -> Iterator[Dict[str, Any]]:
    """Stream CSV rows."""
    if not path.exists():
        return
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            yield dict(row)


def iter_jsonl_rows(path: Path) -> Iterator[Dict[str, Any]]:
    """Stream JSONL rows."""
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            data = json.loads(line)
            if isinstance(data, dict):
                yield data


def iter_table_rows(path: Path) -> Iterator[Dict[str, Any]]:
    """Stream rows from CSV or JSONL."""
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        for row in iter_jsonl_rows(path):
            yield row
    elif suffix == ".csv":
        for row in iter_csv_rows(path):
            yield row


def iter_table_rows_progress(path: Path, show_progress: bool, desc: str, every: int = 100000) -> Iterator[Dict[str, Any]]:
    """Stream rows and print periodic progress for very large files."""
    for idx, row in enumerate(iter_table_rows(path), start=1):
        if show_progress and (idx == 1 or idx % every == 0):
            print("%s %s rows=%d" % (desc, path.name, idx))
        yield row


def find_data_files(root: Path, suffixes: Optional[List[str]] = None, required_name_parts: Optional[List[str]] = None) -> List[Path]:
    """Find likely data files under a root, skipping report/summary folders."""
    if suffixes is None:
        suffixes = [".csv", ".jsonl"]
    if not root.exists():
        return []
    files = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in suffixes:
            continue
        parts = set(path.parts)
        if any(item in parts for item in SKIP_DIR_NAMES):
            continue
        if required_name_parts is not None:
            name = path.name.lower()
            if not any(part.lower() in name for part in required_name_parts):
                continue
        files.append(path)
    return files


def progress_iter(values: Iterable[Any], show_progress: bool, desc: str) -> Iterable[Any]:
    """Use tqdm when available, otherwise yield with lightweight periodic prints."""
    if not show_progress:
        return values
    try:
        from tqdm import tqdm

        return tqdm(values, desc=desc)
    except Exception:
        return _print_progress_iter(values, desc)


def _print_progress_iter(values: Iterable[Any], desc: str) -> Iterator[Any]:
    for idx, item in enumerate(values, start=1):
        if idx == 1 or idx % 25 == 0:
            print("%s: %d" % (desc, idx))
        yield item


def safe_int(value: Any, default: int = 0) -> int:
    """Convert value to int safely."""
    if value in (None, ""):
        return default
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def safe_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    """Convert value to float safely."""
    if value in (None, ""):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def safe_bool(value: Any) -> bool:
    """Convert common bool encodings."""
    if isinstance(value, bool):
        return value
    return str(value).lower() in ("true", "1", "yes")


def safe_json(value: Any, default: Any) -> Any:
    """Parse a JSON field, returning default on failure."""
    if value in (None, ""):
        return default
    if isinstance(value, (dict, list)):
        return value
    try:
        return json.loads(str(value))
    except (TypeError, ValueError):
        return default


def rate(numerator: Any, denominator: Any) -> Optional[float]:
    """Return numerator / denominator or None for invalid denominators."""
    try:
        denom = float(denominator)
        if denom <= 0.0:
            return None
        return float(numerator) / denom
    except (TypeError, ValueError):
        return None


def add_count(counter: Dict[str, int], key: Any, amount: int = 1) -> None:
    """Increment a string-keyed counter."""
    name = str(key) if key not in (None, "") else "unknown"
    counter[name] = counter.get(name, 0) + int(amount)


def merge_count_dict(target: Dict[str, int], source: Any) -> None:
    """Merge string-keyed counts into target."""
    if not isinstance(source, dict):
        return
    for key, value in source.items():
        add_count(target, key, safe_int(value))
