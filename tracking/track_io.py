"""CSV and JSONL I/O for local track records."""

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from deep_oc_sort_3d.tracking.track_types import LocalTrackRecord, array_to_list, list_to_array


CSV_FIELDS = [
    "scene_id",
    "scene_name",
    "split",
    "camera_id",
    "frame_id",
    "local_track_id",
    "detection_id",
    "class_id",
    "class_name",
    "confidence",
    "x1",
    "y1",
    "x2",
    "y2",
    "w",
    "h",
    "center_x",
    "center_y",
    "center_z",
    "width_3d",
    "length_3d",
    "height_3d",
    "yaw",
    "matched_gt_object_id",
    "matched_gt",
    "track_age",
    "track_hits",
    "track_misses",
    "track_state",
]


def write_local_tracks_csv(records: List[LocalTrackRecord], path: Path) -> None:
    """Write local tracks as CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for record in records:
            writer.writerow(track_record_to_csv_row(record))


def read_local_tracks_csv(path: Path) -> List[LocalTrackRecord]:
    """Read local tracks from CSV."""
    if not path.exists():
        return []
    records = []
    with path.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            records.append(track_record_from_csv_row(row))
    return records


def write_local_tracks_jsonl(records: List[LocalTrackRecord], path: Path) -> None:
    """Write local tracks as JSONL."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(track_record_to_dict(record), sort_keys=True) for record in records]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def read_local_tracks_jsonl(path: Path) -> List[LocalTrackRecord]:
    """Read local tracks from JSONL."""
    if not path.exists():
        return []
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        records.append(track_record_from_dict(json.loads(line)))
    return records


def track_record_to_dict(record: LocalTrackRecord) -> Dict[str, Any]:
    """Convert LocalTrackRecord to JSON-friendly dictionary."""
    return {
        "scene_id": record.scene_id,
        "scene_name": record.scene_name,
        "split": record.split,
        "camera_id": record.camera_id,
        "frame_id": record.frame_id,
        "local_track_id": record.local_track_id,
        "detection_id": record.detection_id,
        "class_id": record.class_id,
        "class_name": record.class_name,
        "confidence": record.confidence,
        "bbox_xyxy": list(record.bbox_xyxy),
        "bbox_xywh": list(record.bbox_xywh),
        "center_3d": array_to_list(record.center_3d),
        "dimensions_3d": array_to_list(record.dimensions_3d),
        "yaw": record.yaw,
        "matched_gt_object_id": record.matched_gt_object_id,
        "matched_gt": record.matched_gt,
        "track_age": record.track_age,
        "track_hits": record.track_hits,
        "track_misses": record.track_misses,
        "track_state": record.track_state,
    }


def track_record_from_dict(data: Dict[str, Any]) -> LocalTrackRecord:
    """Create LocalTrackRecord from dictionary."""
    return LocalTrackRecord(
        scene_id=int(data["scene_id"]),
        scene_name=str(data["scene_name"]),
        split=str(data["split"]),
        camera_id=str(data["camera_id"]),
        frame_id=int(data["frame_id"]),
        local_track_id=int(data["local_track_id"]),
        detection_id=int(data["detection_id"]),
        class_id=int(data["class_id"]),
        class_name=str(data["class_name"]),
        confidence=float(data["confidence"]),
        bbox_xyxy=tuple(float(value) for value in data["bbox_xyxy"]),
        bbox_xywh=tuple(float(value) for value in data["bbox_xywh"]),
        center_3d=list_to_array(data.get("center_3d")),
        dimensions_3d=list_to_array(data.get("dimensions_3d")),
        yaw=None if data.get("yaw") is None else float(data["yaw"]),
        matched_gt_object_id=None if data.get("matched_gt_object_id") is None else int(data["matched_gt_object_id"]),
        matched_gt=bool(data.get("matched_gt")),
        track_age=int(data["track_age"]),
        track_hits=int(data["track_hits"]),
        track_misses=int(data["track_misses"]),
        track_state=str(data["track_state"]),
    )


def track_record_to_csv_row(record: LocalTrackRecord) -> Dict[str, Any]:
    """Convert LocalTrackRecord to CSV row."""
    x1, y1, x2, y2 = record.bbox_xyxy
    _x, _y, width, height = record.bbox_xywh
    center = _array_values(record.center_3d, 3)
    dims = _array_values(record.dimensions_3d, 3)
    return {
        "scene_id": record.scene_id,
        "scene_name": record.scene_name,
        "split": record.split,
        "camera_id": record.camera_id,
        "frame_id": record.frame_id,
        "local_track_id": record.local_track_id,
        "detection_id": record.detection_id,
        "class_id": record.class_id,
        "class_name": record.class_name,
        "confidence": record.confidence,
        "x1": x1,
        "y1": y1,
        "x2": x2,
        "y2": y2,
        "w": width,
        "h": height,
        "center_x": _optional_value(center, 0),
        "center_y": _optional_value(center, 1),
        "center_z": _optional_value(center, 2),
        "width_3d": _optional_value(dims, 0),
        "length_3d": _optional_value(dims, 1),
        "height_3d": _optional_value(dims, 2),
        "yaw": "" if record.yaw is None else record.yaw,
        "matched_gt_object_id": "" if record.matched_gt_object_id is None else record.matched_gt_object_id,
        "matched_gt": record.matched_gt,
        "track_age": record.track_age,
        "track_hits": record.track_hits,
        "track_misses": record.track_misses,
        "track_state": record.track_state,
    }


def track_record_from_csv_row(row: Dict[str, str]) -> LocalTrackRecord:
    """Create LocalTrackRecord from CSV row."""
    center = _optional_array([row.get("center_x"), row.get("center_y"), row.get("center_z")])
    dims = _optional_array([row.get("width_3d"), row.get("length_3d"), row.get("height_3d")])
    return LocalTrackRecord(
        scene_id=int(row["scene_id"]),
        scene_name=str(row["scene_name"]),
        split=str(row["split"]),
        camera_id=str(row["camera_id"]),
        frame_id=int(row["frame_id"]),
        local_track_id=int(row["local_track_id"]),
        detection_id=int(row["detection_id"]),
        class_id=int(row["class_id"]),
        class_name=str(row["class_name"]),
        confidence=float(row["confidence"]),
        bbox_xyxy=(float(row["x1"]), float(row["y1"]), float(row["x2"]), float(row["y2"])),
        bbox_xywh=(float(row["x1"]), float(row["y1"]), float(row["w"]), float(row["h"])),
        center_3d=center,
        dimensions_3d=dims,
        yaw=_optional_float(row.get("yaw")),
        matched_gt_object_id=_optional_int(row.get("matched_gt_object_id")),
        matched_gt=str(row.get("matched_gt", "")).lower() in ("true", "1", "yes"),
        track_age=int(row["track_age"]),
        track_hits=int(row["track_hits"]),
        track_misses=int(row["track_misses"]),
        track_state=str(row["track_state"]),
    )


def _array_values(value: Optional[np.ndarray], size: int) -> Optional[np.ndarray]:
    if value is None:
        return None
    arr = np.asarray(value, dtype=float).reshape(-1)
    if arr.size < size:
        return None
    return arr


def _optional_value(values: Optional[np.ndarray], index: int) -> Any:
    if values is None:
        return ""
    return float(values[index])


def _optional_float(value: Any) -> Optional[float]:
    if value in (None, ""):
        return None
    return float(value)


def _optional_int(value: Any) -> Optional[int]:
    if value in (None, ""):
        return None
    return int(float(value))


def _optional_array(values: List[Any]) -> Optional[np.ndarray]:
    if any(value in (None, "") for value in values):
        return None
    return np.asarray([float(value) for value in values], dtype=float)
