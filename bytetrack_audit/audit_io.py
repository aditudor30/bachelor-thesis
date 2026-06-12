"""Small streaming I/O helpers used by the Step 21D audit."""

import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence

import yaml


def read_json(path: Path) -> Dict[str, Any]:
    """Read a JSON mapping or return an empty mapping."""
    if not path.exists():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else {}


def write_json(path: Path, value: Any) -> None:
    """Write deterministic JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str), encoding="utf-8")


def write_yaml(path: Path, value: Dict[str, Any]) -> None:
    """Write YAML without sorting user-facing keys."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")


def read_csv(path: Path) -> List[Dict[str, Any]]:
    """Read a small CSV table."""
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def iter_csv(path: Path) -> Iterator[Dict[str, Any]]:
    """Iterate a CSV without retaining the complete file."""
    if not path.exists():
        return
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            yield dict(row)


def iter_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    """Iterate valid JSON objects in a JSONL file."""
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except ValueError:
                continue
            if isinstance(value, dict):
                yield value


def write_csv(path: Path, rows: Sequence[Dict[str, Any]], fields: Optional[Sequence[str]] = None) -> None:
    """Write a sequence of mappings to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(fields or field_union(rows))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


class CsvSink:
    """Incremental CSV writer for potentially large diagnostic tables."""

    def __init__(self, path: Path, fields: Sequence[str]) -> None:
        self.path = path
        self.fields = list(fields)
        self.handle = None
        self.writer = None

    def __enter__(self) -> "CsvSink":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self.path.open("w", newline="", encoding="utf-8")
        self.writer = csv.DictWriter(self.handle, fieldnames=self.fields, extrasaction="ignore")
        self.writer.writeheader()
        return self

    def write(self, row: Dict[str, Any]) -> None:
        """Write one row."""
        if self.writer is None:
            raise RuntimeError("CsvSink is not open")
        self.writer.writerow({field: row.get(field, "") for field in self.fields})

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        if self.handle is not None:
            self.handle.close()


def progress_iter(values: Sequence[Any], enabled: bool, description: str) -> Iterable[Any]:
    """Use tqdm when available and periodic printing otherwise."""
    if enabled:
        try:
            from tqdm import tqdm

            return tqdm(values, desc=description)
        except ImportError:
            pass
    return _print_progress(values, enabled, description)


def _print_progress(values: Sequence[Any], enabled: bool, description: str) -> Iterable[Any]:
    total = len(values)
    interval = max(1, total // 20) if total else 1
    for index, value in enumerate(values):
        if enabled and (index == 0 or index + 1 == total or (index + 1) % interval == 0):
            print("%s: %d/%d" % (description, index + 1, total))
        yield value


def field_union(rows: Sequence[Dict[str, Any]]) -> List[str]:
    """Return keys in first-seen order."""
    output = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                output.append(str(key))
    return output


def safe_int(value: Any, default: int = 0) -> int:
    """Convert common CSV values to int."""
    if value in (None, ""):
        return default
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def safe_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    """Convert common CSV values to float."""
    if value in (None, ""):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def safe_bool(value: Any) -> bool:
    """Convert common serialized booleans."""
    return str(value).lower() in ("true", "1", "yes")


def count_nonempty_lines(path: Path) -> int:
    """Count non-empty lines incrementally."""
    if not path.exists():
        return 0
    count = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                count += 1
    return count

