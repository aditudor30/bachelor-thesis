"""Small deterministic I/O helpers for Step 22B."""

import csv
import json
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import yaml


def write_json(path: Path, value: Any) -> None:
    """Write deterministic JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str), encoding="utf-8")


def read_json(path: Path) -> Dict[str, Any]:
    """Read a JSON mapping, returning an empty mapping when unavailable."""
    if not path.exists():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else {}


def write_csv(path: Path, rows: Sequence[Dict[str, Any]], fields: Optional[Sequence[str]] = None) -> None:
    """Write dictionaries as CSV with stable columns."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(fields or _collect_fields(rows))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _csv_value(row.get(field)) for field in fieldnames})


def write_yaml(path: Path, value: Dict[str, Any]) -> None:
    """Write a resolved YAML configuration."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")


def prepare_directory(path: Path, overwrite: bool, skip_existing: bool = False) -> bool:
    """Prepare an output directory and report whether work should proceed."""
    if path.exists() and skip_existing:
        return False
    if path.exists() and not overwrite:
        raise FileExistsError("Output already exists; use --overwrite or --skip-existing: %s" % path)
    if path.exists() and overwrite:
        shutil.rmtree(str(path))
    path.mkdir(parents=True, exist_ok=True)
    return True


def progress_iter(values: Sequence[Any], enabled: bool, description: str) -> Iterable[Any]:
    """Use tqdm when installed, with a periodic-print fallback."""
    if enabled:
        try:
            from tqdm import tqdm

            return tqdm(values, desc=description)
        except ImportError:
            pass
    return _fallback_progress(values, enabled, description)


def existing_paths(paths: Sequence[Path]) -> List[Path]:
    """Return unique existing paths in preference order."""
    output = []
    seen = set()
    for path in paths:
        normalized = str(path)
        if normalized not in seen and path.exists():
            output.append(path)
            seen.add(normalized)
    return output


def _collect_fields(rows: Sequence[Dict[str, Any]]) -> List[str]:
    fields = []
    for row in rows:
        for key in row.keys():
            if key not in fields:
                fields.append(str(key))
    return fields or ["status"]


def _csv_value(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, sort_keys=True)
    if value is None:
        return ""
    return value


def _fallback_progress(values: Sequence[Any], enabled: bool, description: str) -> Iterable[Any]:
    total = len(values)
    interval = max(1, total // 20) if total else 1
    for index, value in enumerate(values):
        if enabled and (index == 0 or index + 1 == total or (index + 1) % interval == 0):
            print("%s: %d/%d" % (description, index + 1, total))
        yield value
