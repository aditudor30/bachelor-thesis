"""I/O and progress helpers for Step 20C."""

import csv
import json
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence

import yaml


OUTPUT_DIRS = (
    "configs",
    "features",
    "diagnostics",
    "sweep_runs",
    "comparison",
    "figures",
)


def prepare_output_root(root: Path, overwrite: bool = False) -> Path:
    """Create an isolated output root, optionally replacing it."""
    if overwrite and root.exists():
        shutil.rmtree(str(root))
    for name in OUTPUT_DIRS:
        (root / name).mkdir(parents=True, exist_ok=True)
    return root


def read_csv_rows(path: Path) -> List[Dict[str, Any]]:
    """Read a CSV file into dictionaries."""
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def write_csv_rows(path: Path, rows: Sequence[Dict[str, Any]], fields: Optional[Sequence[str]] = None) -> None:
    """Write rows while preserving a stable union of fields."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(fields or _field_union(rows))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key)) for key in fieldnames})


def read_json(path: Path) -> Optional[Dict[str, Any]]:
    """Read a JSON object when available."""
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    return value if isinstance(value, dict) else None


def write_json(path: Path, value: Any) -> None:
    """Write deterministic JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, default=_json_default)


def write_yaml(path: Path, value: Dict[str, Any]) -> None:
    """Write YAML configuration."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(value, handle, sort_keys=False)


def progress_iter(values: Iterable[Any], enabled: bool, description: str) -> Iterator[Any]:
    """Use tqdm when installed, otherwise print periodic progress."""
    sequence = values if isinstance(values, list) else list(values)
    if enabled:
        try:
            from tqdm import tqdm

            for value in tqdm(sequence, desc=description):
                yield value
            return
        except ImportError:
            pass
    total = len(sequence)
    interval = max(1, total // 20) if total else 1
    for index, value in enumerate(sequence):
        if enabled and (index == 0 or (index + 1) % interval == 0 or index + 1 == total):
            print("%s: %d/%d" % (description, index + 1, total))
        yield value


def safe_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    """Convert a finite numeric value."""
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    if result != result or result in (float("inf"), float("-inf")):
        return default
    return result


def safe_int(value: Any, default: Optional[int] = None) -> Optional[int]:
    """Convert an integer-like value."""
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _field_union(rows: Sequence[Dict[str, Any]]) -> List[str]:
    result = []  # type: List[str]
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                result.append(str(key))
    return result


def _csv_value(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, sort_keys=True)
    if value is None:
        return ""
    return value


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        return value.item()
    raise TypeError("Not JSON serializable: %r" % (value,))
