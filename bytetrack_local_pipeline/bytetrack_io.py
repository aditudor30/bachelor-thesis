"""Small JSON, CSV, YAML and progress helpers for Step 21B."""

import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import yaml


def read_json(path: Path) -> Dict[str, Any]:
    """Read a JSON mapping or return an empty mapping."""
    if not Path(path).exists():
        return {}
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else {}


def write_json(path: Path, value: Any) -> None:
    """Write stable pretty JSON."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, default=str), encoding="utf-8")


def write_yaml(path: Path, value: Dict[str, Any]) -> None:
    """Write YAML without sorting keys."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")


def write_csv(path: Path, rows: Sequence[Dict[str, Any]], fields: Optional[Sequence[str]] = None) -> None:
    """Write dictionaries to CSV."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(fields or _field_union(rows))
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def progress_iter(values: Sequence[Any], enabled: bool, description: str) -> Iterable[Any]:
    """Use tqdm when present, otherwise print periodic progress."""
    if enabled:
        try:
            from tqdm import tqdm

            return tqdm(values, desc=description)
        except ImportError:
            pass
    return _fallback_progress(values, enabled, description)


def _fallback_progress(values: Sequence[Any], enabled: bool, description: str) -> Iterable[Any]:
    total = len(values)
    for index, value in enumerate(values):
        if enabled and (index == 0 or index + 1 == total or (index + 1) % max(1, total // 20) == 0):
            print("%s: %d/%d" % (description, index + 1, total))
        yield value


def _field_union(rows: Sequence[Dict[str, Any]]) -> List[str]:
    fields = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                fields.append(str(key))
                seen.add(key)
    return fields
