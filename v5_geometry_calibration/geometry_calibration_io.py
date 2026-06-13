"""I/O and numeric helpers for V5 calibration."""

import csv
import json
import math
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np
import yaml

from deep_oc_sort_3d.v4_geometry_refinement.track1_geometry_io import (
    clone_row,
    dimensions,
    group_tracks,
    identity_key_set,
    position,
    read_geometry_rows,
    unique_track_count,
    write_variant_rows,
)


def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    """Yield JSON objects from a JSONL file without loading it fully."""
    if not path.is_file():
        return
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            value = json.loads(line)
            if isinstance(value, dict):
                yield value


def read_csv(path: Path) -> List[Dict[str, Any]]:
    """Read a small calibration CSV."""
    if not path.is_file():
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    """Write dictionaries with stable discovered columns."""
    fields: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fields:
                fields.append(str(key))
    if not fields:
        fields = ["status"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key)) for key in fields})


def write_json(path: Path, value: Any) -> None:
    """Write deterministic JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str), encoding="utf-8")


def read_json(path: Path) -> Dict[str, Any]:
    """Read a JSON mapping or return an empty mapping."""
    if not path.is_file():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else {}


def write_yaml(path: Path, value: Dict[str, Any]) -> None:
    """Write resolved configuration YAML."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")


def prepare_directory(path: Path, overwrite: bool, skip_existing: bool = False) -> bool:
    """Prepare an isolated output directory."""
    if path.exists() and skip_existing:
        return False
    if path.exists() and not overwrite:
        raise FileExistsError("Output exists; use --overwrite or --skip-existing: %s" % path)
    if path.exists():
        shutil.rmtree(str(path))
    path.mkdir(parents=True, exist_ok=True)
    return True


def progress_iter(values: Sequence[Any], enabled: bool, description: str) -> Iterable[Any]:
    """Use tqdm when installed, with periodic-print fallback."""
    if enabled:
        try:
            from tqdm import tqdm

            return tqdm(values, desc=description)
        except ImportError:
            pass
    return _fallback_progress(values, enabled, description)


def vector3(value: Any) -> Optional[np.ndarray]:
    """Parse one finite three-vector."""
    try:
        array = np.asarray(value, dtype=float).reshape(-1)
    except (TypeError, ValueError):
        return None
    if array.size < 3 or not np.all(np.isfinite(array[:3])):
        return None
    return array[:3]


def float_value(value: Any) -> Optional[float]:
    """Parse one finite float."""
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def angle_delta(a: float, b: float) -> float:
    """Return signed shortest angular difference b-a."""
    return float((float(b) - float(a) + math.pi) % (2.0 * math.pi) - math.pi)


def normalize_angle(value: float) -> float:
    """Normalize yaw to [-pi, pi]."""
    return float((float(value) + math.pi) % (2.0 * math.pi) - math.pi)


def _csv_value(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, sort_keys=True)
    return "" if value is None else value


def _fallback_progress(values: Sequence[Any], enabled: bool, description: str) -> Iterable[Any]:
    total = len(values)
    interval = max(1, total // 20) if total else 1
    for index, value in enumerate(values):
        if enabled and (index == 0 or index + 1 == total or (index + 1) % interval == 0):
            print("%s: %d/%d" % (description, index + 1, total))
        yield value


__all__ = [
    "angle_delta", "clone_row", "dimensions", "float_value", "group_tracks", "identity_key_set",
    "iter_jsonl", "normalize_angle", "position", "prepare_directory", "progress_iter", "read_csv",
    "read_geometry_rows", "read_json", "unique_track_count", "vector3", "write_csv", "write_json",
    "write_variant_rows", "write_yaml",
]
