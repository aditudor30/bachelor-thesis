"""Dataclasses for global MTMC association outputs."""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class GlobalAssociationEdge:
    """One candidate-to-candidate association edge."""

    scene_name: str
    subset: str
    class_id: int
    class_name: str
    candidate_id_a: str
    candidate_id_b: str
    camera_id_a: str
    camera_id_b: str
    start_frame_a: int
    end_frame_a: int
    start_frame_b: int
    end_frame_b: int
    temporal_relation: str
    overlap_frames: int
    temporal_gap: int
    mean_3d_distance: Optional[float]
    median_3d_distance: Optional[float]
    min_3d_distance: Optional[float]
    max_3d_distance: Optional[float]
    entry_exit_distance: Optional[float]
    velocity_angle_difference: Optional[float]
    cost: float
    affinity: float
    accepted: bool
    reject_reason: str


@dataclass
class GlobalTrack:
    """One global multi-camera track built from local candidates."""

    global_track_id: int
    scene_name: str
    subset: str
    split: str
    class_id: int
    class_name: str
    candidate_ids: List[str]
    camera_ids: List[str]
    local_track_ids: List[int]
    start_frame: int
    end_frame: int
    duration: int
    num_candidates: int
    num_cameras: int
    mean_confidence: float
    max_confidence: float
    trajectory_3d_sampled: List[Tuple[int, float, float, float]]
    center_3d_mean: Optional[np.ndarray]
    majority_gt_object_id: Optional[int]
    gt_purity: Optional[float]
    num_gt_ids: int
    gt_id_counts: Dict[str, int]
    notes: str


def edge_to_dict(edge: GlobalAssociationEdge) -> Dict[str, Any]:
    """Convert an edge to a JSON-friendly dictionary."""
    return {
        "scene_name": edge.scene_name,
        "subset": edge.subset,
        "class_id": edge.class_id,
        "class_name": edge.class_name,
        "candidate_id_a": edge.candidate_id_a,
        "candidate_id_b": edge.candidate_id_b,
        "camera_id_a": edge.camera_id_a,
        "camera_id_b": edge.camera_id_b,
        "start_frame_a": edge.start_frame_a,
        "end_frame_a": edge.end_frame_a,
        "start_frame_b": edge.start_frame_b,
        "end_frame_b": edge.end_frame_b,
        "temporal_relation": edge.temporal_relation,
        "overlap_frames": edge.overlap_frames,
        "temporal_gap": edge.temporal_gap,
        "mean_3d_distance": edge.mean_3d_distance,
        "median_3d_distance": edge.median_3d_distance,
        "min_3d_distance": edge.min_3d_distance,
        "max_3d_distance": edge.max_3d_distance,
        "entry_exit_distance": edge.entry_exit_distance,
        "velocity_angle_difference": edge.velocity_angle_difference,
        "cost": edge.cost,
        "affinity": edge.affinity,
        "accepted": edge.accepted,
        "reject_reason": edge.reject_reason,
    }


def edge_from_dict(data: Dict[str, Any]) -> GlobalAssociationEdge:
    """Create an edge from a dictionary."""
    return GlobalAssociationEdge(
        scene_name=str(data.get("scene_name", "")),
        subset=str(data.get("subset", "")),
        class_id=int(data.get("class_id", -1)),
        class_name=str(data.get("class_name", "")),
        candidate_id_a=str(data.get("candidate_id_a", "")),
        candidate_id_b=str(data.get("candidate_id_b", "")),
        camera_id_a=str(data.get("camera_id_a", "")),
        camera_id_b=str(data.get("camera_id_b", "")),
        start_frame_a=int(data.get("start_frame_a", -1)),
        end_frame_a=int(data.get("end_frame_a", -1)),
        start_frame_b=int(data.get("start_frame_b", -1)),
        end_frame_b=int(data.get("end_frame_b", -1)),
        temporal_relation=str(data.get("temporal_relation", "")),
        overlap_frames=int(data.get("overlap_frames", 0)),
        temporal_gap=int(data.get("temporal_gap", 0)),
        mean_3d_distance=_optional_float(data.get("mean_3d_distance")),
        median_3d_distance=_optional_float(data.get("median_3d_distance")),
        min_3d_distance=_optional_float(data.get("min_3d_distance")),
        max_3d_distance=_optional_float(data.get("max_3d_distance")),
        entry_exit_distance=_optional_float(data.get("entry_exit_distance")),
        velocity_angle_difference=_optional_float(data.get("velocity_angle_difference")),
        cost=float(data.get("cost", 1e9)),
        affinity=float(data.get("affinity", 0.0)),
        accepted=_optional_bool(data.get("accepted", False)),
        reject_reason=str(data.get("reject_reason", "")),
    )


def global_track_to_dict(track: GlobalTrack) -> Dict[str, Any]:
    """Convert a global track to a JSON-friendly dictionary."""
    return {
        "global_track_id": track.global_track_id,
        "scene_name": track.scene_name,
        "subset": track.subset,
        "split": track.split,
        "class_id": track.class_id,
        "class_name": track.class_name,
        "candidate_ids": [str(item) for item in track.candidate_ids],
        "camera_ids": [str(item) for item in track.camera_ids],
        "local_track_ids": [int(item) for item in track.local_track_ids],
        "start_frame": track.start_frame,
        "end_frame": track.end_frame,
        "duration": track.duration,
        "num_candidates": track.num_candidates,
        "num_cameras": track.num_cameras,
        "mean_confidence": track.mean_confidence,
        "max_confidence": track.max_confidence,
        "trajectory_3d_sampled": [list(item) for item in track.trajectory_3d_sampled],
        "center_3d_mean": _array_to_list(track.center_3d_mean),
        "majority_gt_object_id": track.majority_gt_object_id,
        "gt_purity": track.gt_purity,
        "num_gt_ids": track.num_gt_ids,
        "gt_id_counts": dict(track.gt_id_counts),
        "notes": track.notes,
    }


def global_track_from_dict(data: Dict[str, Any]) -> GlobalTrack:
    """Create a global track from a dictionary."""
    return GlobalTrack(
        global_track_id=int(data.get("global_track_id", -1)),
        scene_name=str(data.get("scene_name", "")),
        subset=str(data.get("subset", "")),
        split=str(data.get("split", "")),
        class_id=int(data.get("class_id", -1)),
        class_name=str(data.get("class_name", "")),
        candidate_ids=[str(item) for item in data.get("candidate_ids", [])],
        camera_ids=[str(item) for item in data.get("camera_ids", [])],
        local_track_ids=[int(item) for item in data.get("local_track_ids", [])],
        start_frame=int(data.get("start_frame", -1)),
        end_frame=int(data.get("end_frame", -1)),
        duration=int(data.get("duration", 0)),
        num_candidates=int(data.get("num_candidates", 0)),
        num_cameras=int(data.get("num_cameras", 0)),
        mean_confidence=float(data.get("mean_confidence", 0.0)),
        max_confidence=float(data.get("max_confidence", 0.0)),
        trajectory_3d_sampled=_trajectory_3d(data.get("trajectory_3d_sampled", [])),
        center_3d_mean=_optional_array(data.get("center_3d_mean")),
        majority_gt_object_id=_optional_int(data.get("majority_gt_object_id")),
        gt_purity=_optional_float(data.get("gt_purity")),
        num_gt_ids=int(data.get("num_gt_ids", 0)),
        gt_id_counts=_dict_int(data.get("gt_id_counts", {})),
        notes=str(data.get("notes", "")),
    )


def _array_to_list(value: Optional[np.ndarray]) -> Optional[List[float]]:
    if value is None:
        return None
    return [float(item) for item in np.asarray(value, dtype=float).reshape(-1)]


def _optional_array(value: Any) -> Optional[np.ndarray]:
    if _none_like(value):
        return None
    arr = np.asarray(value, dtype=float).reshape(-1)
    if arr.size == 0:
        return None
    return arr


def _optional_float(value: Any) -> Optional[float]:
    if _none_like(value):
        return None
    return float(value)


def _optional_int(value: Any) -> Optional[int]:
    if _none_like(value):
        return None
    return int(float(value))


def _optional_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).lower() in ("true", "1", "yes")


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


def _none_like(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value == ""
    return False
