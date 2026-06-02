"""Dataclasses and serialization helpers for local tracklets."""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class LocalTracklet:
    """Camera-level local tracklet built from frame-level LocalTrackRecord rows."""

    scene_id: int
    scene_name: str
    split: str
    camera_id: str
    local_track_id: int
    class_id: int
    class_name: str
    start_frame: int
    end_frame: int
    length: int
    frame_ids: List[int]
    detection_ids: List[int]
    mean_confidence: float
    median_confidence: float
    max_confidence: float
    bbox_start: Optional[Tuple[float, float, float, float]]
    bbox_end: Optional[Tuple[float, float, float, float]]
    bbox_mean: Optional[Tuple[float, float, float, float]]
    center_3d_start: Optional[np.ndarray]
    center_3d_end: Optional[np.ndarray]
    center_3d_mean: Optional[np.ndarray]
    center_3d_median: Optional[np.ndarray]
    dimensions_3d_mean: Optional[np.ndarray]
    yaw_mean: Optional[float]
    trajectory_2d: List[Tuple[int, float, float, float, float]]
    trajectory_3d: List[Tuple[int, float, float, float]]
    majority_gt_object_id: Optional[int]
    gt_purity: Optional[float]
    num_gt_ids: int
    gt_id_counts: Dict[str, int]
    quality_score: float
    quality_flag: str
    is_valid_for_mtmc: bool
    notes: str


def local_tracklet_to_dict(tracklet: LocalTracklet) -> Dict[str, Any]:
    """Convert a LocalTracklet into a JSON-friendly dictionary."""
    return {
        "scene_id": tracklet.scene_id,
        "scene_name": tracklet.scene_name,
        "split": tracklet.split,
        "camera_id": tracklet.camera_id,
        "local_track_id": tracklet.local_track_id,
        "class_id": tracklet.class_id,
        "class_name": tracklet.class_name,
        "start_frame": tracklet.start_frame,
        "end_frame": tracklet.end_frame,
        "length": tracklet.length,
        "frame_ids": [int(item) for item in tracklet.frame_ids],
        "detection_ids": [int(item) for item in tracklet.detection_ids],
        "mean_confidence": tracklet.mean_confidence,
        "median_confidence": tracklet.median_confidence,
        "max_confidence": tracklet.max_confidence,
        "bbox_start": _tuple_to_list(tracklet.bbox_start),
        "bbox_end": _tuple_to_list(tracklet.bbox_end),
        "bbox_mean": _tuple_to_list(tracklet.bbox_mean),
        "center_3d_start": _array_to_list(tracklet.center_3d_start),
        "center_3d_end": _array_to_list(tracklet.center_3d_end),
        "center_3d_mean": _array_to_list(tracklet.center_3d_mean),
        "center_3d_median": _array_to_list(tracklet.center_3d_median),
        "dimensions_3d_mean": _array_to_list(tracklet.dimensions_3d_mean),
        "yaw_mean": tracklet.yaw_mean,
        "trajectory_2d": [list(item) for item in tracklet.trajectory_2d],
        "trajectory_3d": [list(item) for item in tracklet.trajectory_3d],
        "majority_gt_object_id": tracklet.majority_gt_object_id,
        "gt_purity": tracklet.gt_purity,
        "num_gt_ids": tracklet.num_gt_ids,
        "gt_id_counts": dict(tracklet.gt_id_counts),
        "quality_score": tracklet.quality_score,
        "quality_flag": tracklet.quality_flag,
        "is_valid_for_mtmc": tracklet.is_valid_for_mtmc,
        "notes": tracklet.notes,
    }


def local_tracklet_from_dict(data: Dict[str, Any]) -> LocalTracklet:
    """Create a LocalTracklet from a dictionary."""
    return LocalTracklet(
        scene_id=int(data.get("scene_id", -1)),
        scene_name=str(data.get("scene_name", "")),
        split=str(data.get("split", "")),
        camera_id=str(data.get("camera_id", "")),
        local_track_id=int(data.get("local_track_id", -1)),
        class_id=int(data.get("class_id", -1)),
        class_name=str(data.get("class_name", "")),
        start_frame=int(data.get("start_frame", -1)),
        end_frame=int(data.get("end_frame", -1)),
        length=int(data.get("length", 0)),
        frame_ids=[int(item) for item in data.get("frame_ids", [])],
        detection_ids=[int(item) for item in data.get("detection_ids", [])],
        mean_confidence=float(data.get("mean_confidence", 0.0)),
        median_confidence=float(data.get("median_confidence", 0.0)),
        max_confidence=float(data.get("max_confidence", 0.0)),
        bbox_start=_optional_tuple4(data.get("bbox_start")),
        bbox_end=_optional_tuple4(data.get("bbox_end")),
        bbox_mean=_optional_tuple4(data.get("bbox_mean")),
        center_3d_start=_optional_array(data.get("center_3d_start")),
        center_3d_end=_optional_array(data.get("center_3d_end")),
        center_3d_mean=_optional_array(data.get("center_3d_mean")),
        center_3d_median=_optional_array(data.get("center_3d_median")),
        dimensions_3d_mean=_optional_array(data.get("dimensions_3d_mean")),
        yaw_mean=_optional_float(data.get("yaw_mean")),
        trajectory_2d=_trajectory_2d(data.get("trajectory_2d", [])),
        trajectory_3d=_trajectory_3d(data.get("trajectory_3d", [])),
        majority_gt_object_id=_optional_int(data.get("majority_gt_object_id")),
        gt_purity=_optional_float(data.get("gt_purity")),
        num_gt_ids=int(data.get("num_gt_ids", 0)),
        gt_id_counts=_gt_counts(data.get("gt_id_counts", {})),
        quality_score=float(data.get("quality_score", 0.0)),
        quality_flag=str(data.get("quality_flag", "invalid")),
        is_valid_for_mtmc=bool(data.get("is_valid_for_mtmc", False)),
        notes=str(data.get("notes", "")),
    )


def _array_to_list(value: Optional[np.ndarray]) -> Optional[List[float]]:
    if value is None:
        return None
    return [float(item) for item in np.asarray(value, dtype=float).reshape(-1)]


def _tuple_to_list(value: Optional[Tuple[float, float, float, float]]) -> Optional[List[float]]:
    if value is None:
        return None
    return [float(item) for item in value]


def _optional_array(value: Any) -> Optional[np.ndarray]:
    if _is_none_like(value):
        return None
    arr = np.asarray(value, dtype=float).reshape(-1)
    if arr.size == 0:
        return None
    return arr


def _optional_tuple4(value: Any) -> Optional[Tuple[float, float, float, float]]:
    if _is_none_like(value):
        return None
    items = list(value)
    if len(items) < 4:
        return None
    return (float(items[0]), float(items[1]), float(items[2]), float(items[3]))


def _optional_float(value: Any) -> Optional[float]:
    if _is_none_like(value):
        return None
    return float(value)


def _optional_int(value: Any) -> Optional[int]:
    if _is_none_like(value):
        return None
    return int(float(value))


def _trajectory_2d(value: Any) -> List[Tuple[int, float, float, float, float]]:
    output = []
    for item in value or []:
        if len(item) < 5:
            continue
        output.append((int(item[0]), float(item[1]), float(item[2]), float(item[3]), float(item[4])))
    return output


def _trajectory_3d(value: Any) -> List[Tuple[int, float, float, float]]:
    output = []
    for item in value or []:
        if len(item) < 4:
            continue
        output.append((int(item[0]), float(item[1]), float(item[2]), float(item[3])))
    return output


def _gt_counts(value: Any) -> Dict[str, int]:
    if not isinstance(value, dict):
        return {}
    return {str(key): int(item) for key, item in value.items()}


def _is_none_like(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value == ""
    return False
