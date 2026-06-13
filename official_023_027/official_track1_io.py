"""Track1 row, JSON, CSV, checksum and progress helpers for Step 22A."""

import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


@dataclass
class OfficialTrack1Row:
    """One official Track1 row."""

    scene_id: int
    class_id: int
    object_id: int
    frame_id: int
    x: float
    y: float
    z: float
    width: float
    length: float
    height: float
    yaw: float
    source_line: int = 0
    confidence: Optional[float] = None

    def key(self) -> Tuple[int, int, int, int]:
        """Return the official duplicate/sort key."""
        return (self.scene_id, self.class_id, self.object_id, self.frame_id)


def read_track1_rows(path: Path, progress: bool = True) -> List[OfficialTrack1Row]:
    """Read an eleven-column Track1 text file."""
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        lines = list(handle)
    for line_number, line in enumerate(progress_iter(lines, progress, "read %s" % path.name), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        parts = stripped.split()
        if len(parts) != 11:
            raise ValueError("Line %d in %s has %d columns, expected 11" % (line_number, path, len(parts)))
        rows.append(
            OfficialTrack1Row(
                scene_id=int(float(parts[0])),
                class_id=int(float(parts[1])),
                object_id=int(float(parts[2])),
                frame_id=int(float(parts[3])),
                x=float(parts[4]),
                y=float(parts[5]),
                z=float(parts[6]),
                width=float(parts[7]),
                length=float(parts[8]),
                height=float(parts[9]),
                yaw=float(parts[10]),
                source_line=line_number,
            )
        )
    return rows


def write_track1_rows(path: Path, rows: Sequence[OfficialTrack1Row], decimals: int = 2) -> int:
    """Write sorted Track1 rows with fixed decimal precision."""
    path.parent.mkdir(parents=True, exist_ok=True)
    template = "%%.%df" % int(decimals)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            values = [
                str(int(row.scene_id)),
                str(int(row.class_id)),
                str(int(row.object_id)),
                str(int(row.frame_id)),
                template % row.x,
                template % row.y,
                template % row.z,
                template % row.width,
                template % row.length,
                template % row.height,
                template % row.yaw,
            ]
            handle.write(" ".join(values) + "\n")
    return len(rows)


def compute_sha256(path: Path) -> str:
    """Compute a file SHA256 incrementally."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def count_nonempty_lines(path: Path) -> int:
    """Count non-empty lines without loading the file."""
    count = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                count += 1
    return count


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


def write_csv(path: Path, rows: Sequence[Dict[str, Any]], fields: Sequence[str]) -> None:
    """Write dictionaries to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def write_yaml(path: Path, value: Dict[str, Any]) -> None:
    """Write a generated YAML configuration."""
    import yaml

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")


def progress_iter(values: Sequence[Any], enabled: bool, description: str) -> Iterable[Any]:
    """Use tqdm when available, otherwise print periodic progress."""
    if enabled:
        try:
            from tqdm import tqdm

            return tqdm(values, desc=description)
        except ImportError:
            pass
    return _fallback_progress(values, enabled, description)


def _fallback_progress(values: Sequence[Any], enabled: bool, description: str) -> Iterable[Any]:
    total = len(values)
    interval = max(1, total // 20) if total else 1
    for index, value in enumerate(values):
        if enabled and (index == 0 or index + 1 == total or (index + 1) % interval == 0):
            print("%s: %d/%d" % (description, index + 1, total))
        yield value
