"""Track1-like row model and parser for Step 23A."""

from dataclasses import dataclass, replace
from pathlib import Path
from typing import List, Optional, Sequence, Tuple


@dataclass
class AuditTrack1Row:
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
    raw_class_id: Optional[int] = None
    source_class_space: str = "official"
    source_path: str = ""
    source_kind: str = "track1"
    confidence: Optional[float] = None
    coordinate_frame: str = "unknown"

    def key(self) -> Tuple[int, int, int, int]:
        return (self.scene_id, self.class_id, self.object_id, self.frame_id)

    def clone(self, **changes: object) -> "AuditTrack1Row":
        return replace(self, **changes)


def read_track1_like(path: Path) -> List[AuditTrack1Row]:
    rows: List[AuditTrack1Row] = []
    if not path.is_file():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            stripped = line.strip()
            if not stripped:
                continue
            parts = stripped.split()
            if len(parts) != 11:
                raise ValueError("Line %d in %s has %d columns, expected 11" % (line_number, path, len(parts)))
            class_id = _int(parts[1])
            rows.append(AuditTrack1Row(
                scene_id=_int(parts[0]), class_id=class_id, object_id=_int(parts[2]),
                frame_id=_int(parts[3]), x=float(parts[4]), y=float(parts[5]), z=float(parts[6]),
                width=float(parts[7]), length=float(parts[8]), height=float(parts[9]), yaw=float(parts[10]),
                raw_class_id=class_id, source_class_space="official", source_path=str(path), source_kind="track1",
            ))
    return rows


def write_track1_like(path: Path, rows: Sequence[AuditTrack1Row], decimals: int = 2) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    template = "%%.%df" % int(decimals)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in sorted(rows, key=lambda item: item.key()):
            values = [
                str(int(row.scene_id)), str(int(row.class_id)), str(int(row.object_id)), str(int(row.frame_id)),
                template % row.x, template % row.y, template % row.z,
                template % row.width, template % row.length, template % row.height, template % row.yaw,
            ]
            handle.write(" ".join(values) + "\n")
    return len(rows)


def _int(value: str) -> int:
    numeric = float(value)
    if not numeric.is_integer():
        raise ValueError("Expected integer token: %s" % value)
    return int(numeric)
