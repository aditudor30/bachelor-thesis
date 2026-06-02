"""Dataclasses and serialization helpers for frame-level global records."""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class GlobalFrameRecord:
    """One frame-level detection/track record with an optional global id."""

    scene_id: int
    scene_name: str
    split: str
    subset: str
    camera_id: str
    frame_id: int
    global_track_id: Optional[int]
    local_track_id: int
    candidate_id: Optional[str]
    detection_id: int
    class_id: int
    class_name: str
    confidence: float
    bbox_xyxy: Tuple[float, float, float, float]
    bbox_xywh: Tuple[float, float, float, float]
    center_3d: Optional[np.ndarray]
    dimensions_3d: Optional[np.ndarray]
    yaw: Optional[float]
    matched_gt_object_id: Optional[int]
    matched_gt: bool
    source: str


def global_frame_record_to_dict(record: GlobalFrameRecord) -> Dict[str, Any]:
    """Convert a GlobalFrameRecord to a JSON-friendly dictionary."""
    return {
        "scene_id": record.scene_id,
        "scene_name": record.scene_name,
        "split": record.split,
        "subset": record.subset,
        "camera_id": record.camera_id,
        "frame_id": record.frame_id,
        "global_track_id": record.global_track_id,
        "local_track_id": record.local_track_id,
        "candidate_id": record.candidate_id,
        "detection_id": record.detection_id,
        "class_id": record.class_id,
        "class_name": record.class_name,
        "confidence": record.confidence,
        "bbox_xyxy": [float(item) for item in record.bbox_xyxy],
        "bbox_xywh": [float(item) for item in record.bbox_xywh],
        "center_3d": _array_to_list(record.center_3d),
        "dimensions_3d": _array_to_list(record.dimensions_3d),
        "yaw": record.yaw,
        "matched_gt_object_id": record.matched_gt_object_id,
        "matched_gt": record.matched_gt,
        "source": record.source,
    }


def global_frame_record_from_dict(data: Dict[str, Any]) -> GlobalFrameRecord:
    """Create a GlobalFrameRecord from a dictionary."""
    return GlobalFrameRecord(
        scene_id=int(data.get("scene_id", -1)),
        scene_name=str(data.get("scene_name", "")),
        split=str(data.get("split", "")),
        subset=str(data.get("subset", "")),
        camera_id=str(data.get("camera_id", "")),
        frame_id=int(data.get("frame_id", -1)),
        global_track_id=_optional_int(data.get("global_track_id")),
        local_track_id=int(data.get("local_track_id", -1)),
        candidate_id=_optional_str(data.get("candidate_id")),
        detection_id=int(data.get("detection_id", -1)),
        class_id=int(data.get("class_id", -1)),
        class_name=str(data.get("class_name", "")),
        confidence=float(data.get("confidence", 0.0)),
        bbox_xyxy=_tuple4(data.get("bbox_xyxy", [0.0, 0.0, 0.0, 0.0])),
        bbox_xywh=_tuple4(data.get("bbox_xywh", [0.0, 0.0, 0.0, 0.0])),
        center_3d=_optional_array(data.get("center_3d")),
        dimensions_3d=_optional_array(data.get("dimensions_3d")),
        yaw=_optional_float(data.get("yaw")),
        matched_gt_object_id=_optional_int(data.get("matched_gt_object_id")),
        matched_gt=_bool(data.get("matched_gt", False)),
        source=str(data.get("source", "")),
    )


def _array_to_list(value: Optional[np.ndarray]) -> Optional[List[float]]:
    if value is None:
        return None
    return [float(item) for item in np.asarray(value, dtype=float).reshape(-1)]


def _optional_array(value: Any) -> Optional[np.ndarray]:
    if value is None:
        return None
    if isinstance(value, str) and value == "":
        return None
    arr = np.asarray(value, dtype=float).reshape(-1)
    if arr.size == 0:
        return None
    return arr


def _tuple4(value: Any) -> Tuple[float, float, float, float]:
    items = list(value)
    if len(items) < 4:
        return (0.0, 0.0, 0.0, 0.0)
    return (float(items[0]), float(items[1]), float(items[2]), float(items[3]))


def _optional_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, str) and value == "":
        return None
    return float(value)


def _optional_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, str) and value == "":
        return None
    return int(float(value))


def _optional_str(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, str) and value == "":
        return None
    return str(value)


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).lower() in ("true", "1", "yes")
