"""Generic configurable frame-level tracking export."""

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from deep_oc_sort_3d.final_export.global_frame_types import (
    GlobalFrameRecord,
    global_frame_record_from_dict,
    global_frame_record_to_dict,
)


FRAME_RECORD_CSV_FIELDS = [
    "scene_id",
    "scene_name",
    "split",
    "subset",
    "camera_id",
    "frame_id",
    "global_track_id",
    "local_track_id",
    "candidate_id",
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
    "source",
]


GENERIC_EXPORT_FIELDS = [
    "scene_name",
    "camera_id",
    "frame_id",
    "global_track_id",
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
]


def write_global_frame_records_csv(records: List[GlobalFrameRecord], path: Path) -> None:
    """Write frame-level global records as CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FRAME_RECORD_CSV_FIELDS)
        writer.writeheader()
        for record in records:
            writer.writerow(global_frame_record_to_csv_row(record))


def read_global_frame_records_csv(path: Path) -> List[GlobalFrameRecord]:
    """Read frame-level global records from CSV."""
    if not path.exists():
        return []
    records = []
    with path.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            records.append(global_frame_record_from_csv_row(row))
    return records


def write_global_frame_records_jsonl(records: List[GlobalFrameRecord], path: Path) -> None:
    """Write frame-level global records as JSONL."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(global_frame_record_to_dict(record), sort_keys=True) for record in records]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def read_global_frame_records_jsonl(path: Path) -> List[GlobalFrameRecord]:
    """Read frame-level global records from JSONL."""
    if not path.exists():
        return []
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(global_frame_record_from_dict(json.loads(line)))
    return records


def read_global_frame_records_file(path: Path) -> List[GlobalFrameRecord]:
    """Read frame-level global records from CSV or JSONL."""
    if path.suffix.lower() == ".jsonl":
        return read_global_frame_records_jsonl(path)
    return read_global_frame_records_csv(path)


def export_generic_tracking_scene_csv(
    frame_record_files: List[Path],
    output_path: Path,
    drop_unassigned: bool = True,
    drop_invalid_bbox: bool = True,
) -> Dict[str, Any]:
    """Export one generic CSV per scene.

    TODO: track1.txt exact format should be implemented once the official
    submission schema is confirmed.
    """
    records = []
    for path in frame_record_files:
        records.extend(read_global_frame_records_file(path))
    if drop_unassigned:
        records = [record for record in records if record.global_track_id is not None]
    if drop_invalid_bbox:
        records = [record for record in records if is_valid_bbox(record.bbox_xyxy)]
    records = sorted(records, key=lambda item: (item.scene_name, item.camera_id, item.frame_id, int(item.global_track_id or -1)))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=GENERIC_EXPORT_FIELDS)
        writer.writeheader()
        for record in records:
            writer.writerow(_generic_export_row(record))
    return _generic_export_summary(records, output_path)


def is_valid_bbox(bbox_xyxy: Any) -> bool:
    """Return True when bbox has positive width and height."""
    try:
        x1, y1, x2, y2 = bbox_xyxy
    except (TypeError, ValueError):
        return False
    return float(x2) > float(x1) and float(y2) > float(y1)


def global_frame_record_to_csv_row(record: GlobalFrameRecord) -> Dict[str, Any]:
    """Convert a GlobalFrameRecord to a CSV row."""
    x1, y1, x2, y2 = record.bbox_xyxy
    _x, _y, width, height = record.bbox_xywh
    center = _array_values(record.center_3d, 3)
    dims = _array_values(record.dimensions_3d, 3)
    return {
        "scene_id": record.scene_id,
        "scene_name": record.scene_name,
        "split": record.split,
        "subset": record.subset,
        "camera_id": record.camera_id,
        "frame_id": record.frame_id,
        "global_track_id": "" if record.global_track_id is None else int(record.global_track_id),
        "local_track_id": record.local_track_id,
        "candidate_id": "" if record.candidate_id is None else record.candidate_id,
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
        "source": record.source,
    }


def global_frame_record_from_csv_row(row: Dict[str, str]) -> GlobalFrameRecord:
    """Create a GlobalFrameRecord from a CSV row."""
    bbox_xyxy = (float(row["x1"]), float(row["y1"]), float(row["x2"]), float(row["y2"]))
    bbox_xywh = (float(row["x1"]), float(row["y1"]), float(row["w"]), float(row["h"]))
    return GlobalFrameRecord(
        scene_id=int(row["scene_id"]),
        scene_name=str(row["scene_name"]),
        split=str(row["split"]),
        subset=str(row["subset"]),
        camera_id=str(row["camera_id"]),
        frame_id=int(row["frame_id"]),
        global_track_id=_optional_int(row.get("global_track_id")),
        local_track_id=int(row["local_track_id"]),
        candidate_id=_optional_str(row.get("candidate_id")),
        detection_id=int(row["detection_id"]),
        class_id=int(row["class_id"]),
        class_name=str(row["class_name"]),
        confidence=float(row["confidence"]),
        bbox_xyxy=bbox_xyxy,
        bbox_xywh=bbox_xywh,
        center_3d=_optional_array([row.get("center_x"), row.get("center_y"), row.get("center_z")]),
        dimensions_3d=_optional_array([row.get("width_3d"), row.get("length_3d"), row.get("height_3d")]),
        yaw=_optional_float(row.get("yaw")),
        matched_gt_object_id=_optional_int(row.get("matched_gt_object_id")),
        matched_gt=_bool(row.get("matched_gt", False)),
        source=str(row.get("source", "")),
    )


def _generic_export_row(record: GlobalFrameRecord) -> Dict[str, Any]:
    row = global_frame_record_to_csv_row(record)
    return {key: row[key] for key in GENERIC_EXPORT_FIELDS}


def _generic_export_summary(records: List[GlobalFrameRecord], output_path: Path) -> Dict[str, Any]:
    assigned = [record for record in records if record.global_track_id is not None]
    frame_ids = [record.frame_id for record in records]
    return {
        "scene_name": records[0].scene_name if records else "",
        "output_path": str(output_path),
        "rows_written": len(records),
        "unique_global_tracks": len(set([record.global_track_id for record in assigned])),
        "cameras": sorted(set([record.camera_id for record in records])),
        "frame_min": min(frame_ids) if frame_ids else None,
        "frame_max": max(frame_ids) if frame_ids else None,
        "per_class_counts": _per_class_counts(records),
    }


def _per_class_counts(records: List[GlobalFrameRecord]) -> Dict[str, int]:
    counts = {}
    for record in records:
        key = str(record.class_name)
        counts[key] = counts.get(key, 0) + 1
    return counts


def _array_values(value: Optional[np.ndarray], size: int) -> Optional[np.ndarray]:
    if value is None:
        return None
    arr = np.asarray(value, dtype=float).reshape(-1)
    if arr.size < size:
        return None
    return arr


def _optional_array(values: List[Any]) -> Optional[np.ndarray]:
    if any(value in (None, "") for value in values):
        return None
    return np.asarray([float(value) for value in values], dtype=float)


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


def _optional_str(value: Any) -> Optional[str]:
    if value in (None, ""):
        return None
    return str(value)


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).lower() in ("true", "1", "yes")
