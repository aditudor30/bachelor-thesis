"""Track1 geometry grouping, copying, change and output helpers."""

import csv
import json
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import yaml

from deep_oc_sort_3d.official_023_027.official_track1_io import OfficialTrack1Row, read_track1_rows, write_track1_rows


TrackKey = Tuple[int, int, int]


def read_geometry_rows(path: Path, progress: bool = True) -> List[OfficialTrack1Row]:
    """Read sorted official Track1 geometry rows."""
    return sorted(read_track1_rows(path, progress=progress), key=lambda row: row.key())


def group_tracks(rows: Sequence[OfficialTrack1Row]) -> Dict[TrackKey, List[OfficialTrack1Row]]:
    """Group rows by scene, official class and immutable object ID."""
    grouped = defaultdict(list)
    for row in rows:
        grouped[(int(row.scene_id), int(row.class_id), int(row.object_id))].append(row)
    return {key: sorted(values, key=lambda row: row.frame_id) for key, values in grouped.items()}


def clone_row(row: OfficialTrack1Row, **changes: Any) -> OfficialTrack1Row:
    """Copy a row while changing geometry fields only."""
    values = {
        "scene_id": row.scene_id, "class_id": row.class_id, "object_id": row.object_id, "frame_id": row.frame_id,
        "x": row.x, "y": row.y, "z": row.z, "width": row.width, "length": row.length,
        "height": row.height, "yaw": row.yaw, "source_line": row.source_line, "confidence": row.confidence,
    }
    for key, value in changes.items():
        if key not in ("x", "y", "z", "width", "length", "height", "yaw"):
            raise ValueError("V4 cannot modify immutable Track1 field: %s" % key)
        values[key] = value
    return OfficialTrack1Row(**values)


def position(row: OfficialTrack1Row) -> np.ndarray:
    """Return xyz as a float array."""
    return np.asarray([row.x, row.y, row.z], dtype=float)


def dimensions(row: OfficialTrack1Row) -> np.ndarray:
    """Return width/length/height as a float array."""
    return np.asarray([row.width, row.length, row.height], dtype=float)


def write_variant_rows(path: Path, rows: Sequence[OfficialTrack1Row], config: Dict[str, Any]) -> int:
    """Write sorted fixed-precision Track1 rows."""
    decimals = int(config.get("official_track1", {}).get("round_float_decimals", 2))
    return write_track1_rows(path, sorted(rows, key=lambda row: row.key()), decimals=decimals)


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


def write_csv(path: Path, rows: Sequence[Dict[str, Any]], fields: Optional[Sequence[str]] = None) -> None:
    """Write dictionaries to CSV with stable field order."""
    fieldnames = list(fields or _fields(rows))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _csv_value(row.get(field)) for field in fieldnames})


def write_yaml(path: Path, value: Dict[str, Any]) -> None:
    """Write resolved YAML."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")


def prepare_directory(path: Path, overwrite: bool, skip_existing: bool = False) -> bool:
    """Prepare isolated output without silently replacing prior results."""
    if path.exists() and skip_existing:
        return False
    if path.exists() and not overwrite:
        raise FileExistsError("Output exists; use --overwrite or --skip-existing: %s" % path)
    if path.exists() and overwrite:
        shutil.rmtree(str(path))
    path.mkdir(parents=True, exist_ok=True)
    return True


def progress_iter(values: Sequence[Any], enabled: bool, description: str) -> Iterable[Any]:
    """Use tqdm when available, otherwise print periodic progress."""
    if enabled:
        try:
            from tqdm import tqdm

            return tqdm(values, desc=description)
        except ImportError:
            pass
    return _fallback_progress(values, enabled, description)


def identity_key_set(rows: Sequence[OfficialTrack1Row]) -> set:
    """Return immutable row keys."""
    return set(row.key() for row in rows)


def unique_track_count(rows: Sequence[OfficialTrack1Row]) -> int:
    """Return unique immutable track count."""
    return len(set((row.scene_id, row.class_id, row.object_id) for row in rows))


def _fields(rows: Sequence[Dict[str, Any]]) -> List[str]:
    output = []
    for row in rows:
        for key in row.keys():
            if key not in output:
                output.append(str(key))
    return output or ["status"]


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
