"""Small I/O and progress helpers for Step 21E."""

import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import yaml


def read_json(path: Path) -> Dict[str, Any]:
    """Read a JSON mapping, returning an empty mapping when unavailable."""
    if not path.exists():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else {}


def read_csv(path: Path) -> List[Dict[str, Any]]:
    """Read a CSV file as dictionaries."""
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_json(path: Path, value: Any) -> None:
    """Write deterministic, human-readable JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str), encoding="utf-8")


def write_yaml(path: Path, value: Dict[str, Any]) -> None:
    """Write YAML while preserving mapping insertion order."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")


def write_csv(
    path: Path,
    rows: Sequence[Dict[str, Any]],
    fields: Optional[Sequence[str]] = None,
) -> None:
    """Write mappings to CSV, preserving the first-seen field order."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(fields or _field_union(rows))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def progress_iter(values: Sequence[Any], enabled: bool, description: str) -> Iterable[Any]:
    """Use tqdm when installed and periodic prints otherwise."""
    if enabled:
        try:
            from tqdm import tqdm

            return tqdm(values, desc=description)
        except ImportError:
            pass
    return _fallback_progress(values, enabled, description)


def _fallback_progress(values: Sequence[Any], enabled: bool, description: str) -> Iterable[Any]:
    total = len(values)
    interval = max(1, total // 20)
    for index, value in enumerate(values):
        if enabled and (index == 0 or index + 1 == total or (index + 1) % interval == 0):
            print("%s: %d/%d" % (description, index + 1, total))
        yield value


def _field_union(rows: Sequence[Dict[str, Any]]) -> List[str]:
    output = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                output.append(str(key))
    return output

