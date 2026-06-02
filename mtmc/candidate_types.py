"""Dataclasses and serialization helpers for MTMC tracklet candidates."""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class MTMCTrackletCandidate:
    """Compact MTMC-ready representation of one local tracklet."""

    candidate_id: str
    scene_id: int
    scene_name: str
    split: str
    subset: str
    camera_id: str
    local_track_id: int
    class_id: int
    class_name: str
    start_frame: int
    end_frame: int
    length: int
    duration: int
    mean_confidence: float
    median_confidence: float
    max_confidence: float
    quality_score: float
    quality_flag: str
    source_tracklet_valid_for_mtmc: bool
    is_candidate: bool
    reject_reason: Optional[str]
    bbox_start: Optional[Tuple[float, float, float, float]]
    bbox_end: Optional[Tuple[float, float, float, float]]
    bbox_mean: Optional[Tuple[float, float, float, float]]
    center_3d_start: Optional[np.ndarray]
    center_3d_end: Optional[np.ndarray]
    center_3d_mean: Optional[np.ndarray]
    center_3d_median: Optional[np.ndarray]
    trajectory_2d_sampled: List[Tuple[int, float, float, float, float]]
    trajectory_3d_sampled: List[Tuple[int, float, float, float]]
    trajectory_3d_length: int
    has_3d: bool
    entry_frame: int
    exit_frame: int
    entry_center_3d: Optional[np.ndarray]
    exit_center_3d: Optional[np.ndarray]
    mean_velocity_3d: Optional[np.ndarray]
    travel_distance_3d: Optional[float]
    majority_gt_object_id: Optional[int]
    gt_purity: Optional[float]
    num_gt_ids: int
    gt_id_counts: Dict[str, int]
    reid_embedding_path: Optional[str]
    reid_embedding: Optional[np.ndarray]
    global_track_id: Optional[int]


def make_candidate_id(scene_name: str, camera_id: str, local_track_id: int) -> str:
    """Return a deterministic candidate id."""
    return "%s__%s__track_%d" % (str(scene_name), str(camera_id), int(local_track_id))


def candidate_to_dict(candidate: MTMCTrackletCandidate) -> Dict[str, Any]:
    """Convert a candidate to a JSON-friendly dictionary."""
    return {
        "candidate_id": candidate.candidate_id,
        "scene_id": candidate.scene_id,
        "scene_name": candidate.scene_name,
        "split": candidate.split,
        "subset": candidate.subset,
        "camera_id": candidate.camera_id,
        "local_track_id": candidate.local_track_id,
        "class_id": candidate.class_id,
        "class_name": candidate.class_name,
        "start_frame": candidate.start_frame,
        "end_frame": candidate.end_frame,
        "length": candidate.length,
        "duration": candidate.duration,
        "mean_confidence": candidate.mean_confidence,
        "median_confidence": candidate.median_confidence,
        "max_confidence": candidate.max_confidence,
        "quality_score": candidate.quality_score,
        "quality_flag": candidate.quality_flag,
        "source_tracklet_valid_for_mtmc": candidate.source_tracklet_valid_for_mtmc,
        "is_candidate": candidate.is_candidate,
        "reject_reason": candidate.reject_reason,
        "bbox_start": _tuple_to_list(candidate.bbox_start),
        "bbox_end": _tuple_to_list(candidate.bbox_end),
        "bbox_mean": _tuple_to_list(candidate.bbox_mean),
        "center_3d_start": _array_to_list(candidate.center_3d_start),
        "center_3d_end": _array_to_list(candidate.center_3d_end),
        "center_3d_mean": _array_to_list(candidate.center_3d_mean),
        "center_3d_median": _array_to_list(candidate.center_3d_median),
        "trajectory_2d_sampled": [list(item) for item in candidate.trajectory_2d_sampled],
        "trajectory_3d_sampled": [list(item) for item in candidate.trajectory_3d_sampled],
        "trajectory_3d_length": candidate.trajectory_3d_length,
        "has_3d": candidate.has_3d,
        "entry_frame": candidate.entry_frame,
        "exit_frame": candidate.exit_frame,
        "entry_center_3d": _array_to_list(candidate.entry_center_3d),
        "exit_center_3d": _array_to_list(candidate.exit_center_3d),
        "mean_velocity_3d": _array_to_list(candidate.mean_velocity_3d),
        "travel_distance_3d": candidate.travel_distance_3d,
        "majority_gt_object_id": candidate.majority_gt_object_id,
        "gt_purity": candidate.gt_purity,
        "num_gt_ids": candidate.num_gt_ids,
        "gt_id_counts": dict(candidate.gt_id_counts),
        "reid_embedding_path": candidate.reid_embedding_path,
        "reid_embedding": _array_to_list(candidate.reid_embedding),
        "global_track_id": candidate.global_track_id,
    }


def candidate_from_dict(data: Dict[str, Any]) -> MTMCTrackletCandidate:
    """Create a candidate from a dictionary."""
    return MTMCTrackletCandidate(
        candidate_id=str(data.get("candidate_id", "")),
        scene_id=int(data.get("scene_id", -1)),
        scene_name=str(data.get("scene_name", "")),
        split=str(data.get("split", "")),
        subset=str(data.get("subset", "")),
        camera_id=str(data.get("camera_id", "")),
        local_track_id=int(data.get("local_track_id", -1)),
        class_id=int(data.get("class_id", -1)),
        class_name=str(data.get("class_name", "")),
        start_frame=int(data.get("start_frame", -1)),
        end_frame=int(data.get("end_frame", -1)),
        length=int(data.get("length", 0)),
        duration=int(data.get("duration", 0)),
        mean_confidence=float(data.get("mean_confidence", 0.0)),
        median_confidence=float(data.get("median_confidence", 0.0)),
        max_confidence=float(data.get("max_confidence", 0.0)),
        quality_score=float(data.get("quality_score", 0.0)),
        quality_flag=str(data.get("quality_flag", "")),
        source_tracklet_valid_for_mtmc=bool(data.get("source_tracklet_valid_for_mtmc", False)),
        is_candidate=bool(data.get("is_candidate", False)),
        reject_reason=_optional_str(data.get("reject_reason")),
        bbox_start=_optional_tuple4(data.get("bbox_start")),
        bbox_end=_optional_tuple4(data.get("bbox_end")),
        bbox_mean=_optional_tuple4(data.get("bbox_mean")),
        center_3d_start=_optional_array(data.get("center_3d_start")),
        center_3d_end=_optional_array(data.get("center_3d_end")),
        center_3d_mean=_optional_array(data.get("center_3d_mean")),
        center_3d_median=_optional_array(data.get("center_3d_median")),
        trajectory_2d_sampled=_trajectory_2d(data.get("trajectory_2d_sampled", [])),
        trajectory_3d_sampled=_trajectory_3d(data.get("trajectory_3d_sampled", [])),
        trajectory_3d_length=int(data.get("trajectory_3d_length", 0)),
        has_3d=bool(data.get("has_3d", False)),
        entry_frame=int(data.get("entry_frame", -1)),
        exit_frame=int(data.get("exit_frame", -1)),
        entry_center_3d=_optional_array(data.get("entry_center_3d")),
        exit_center_3d=_optional_array(data.get("exit_center_3d")),
        mean_velocity_3d=_optional_array(data.get("mean_velocity_3d")),
        travel_distance_3d=_optional_float(data.get("travel_distance_3d")),
        majority_gt_object_id=_optional_int(data.get("majority_gt_object_id")),
        gt_purity=_optional_float(data.get("gt_purity")),
        num_gt_ids=int(data.get("num_gt_ids", 0)),
        gt_id_counts=_dict_int(data.get("gt_id_counts", {})),
        reid_embedding_path=_optional_str(data.get("reid_embedding_path")),
        reid_embedding=_optional_array(data.get("reid_embedding")),
        global_track_id=_optional_int(data.get("global_track_id")),
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


def _optional_str(value: Any) -> Optional[str]:
    if _is_none_like(value):
        return None
    return str(value)


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


def _dict_int(value: Any) -> Dict[str, int]:
    if not isinstance(value, dict):
        return {}
    return {str(key): int(item) for key, item in value.items()}


def _is_none_like(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value == ""
    return False
