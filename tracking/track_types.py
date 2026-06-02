"""Dataclasses and conversion helpers for local single-camera tracks."""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from deep_oc_sort_3d.observations.observation_types import Observation3D


@dataclass
class LocalTrackDetection:
    """One local tracker input detection converted from Observation3D."""

    scene_id: int
    scene_name: str
    split: str
    camera_id: str
    frame_id: int
    detection_id: int
    class_id: int
    class_name: str
    confidence: float
    bbox_xyxy: Tuple[float, float, float, float]
    bbox_xywh: Tuple[float, float, float, float]
    center_3d: Optional[np.ndarray]
    dimensions_3d: Optional[np.ndarray]
    yaw: Optional[float]
    object_id: Optional[int]
    matched_gt: bool
    matched_iou: Optional[float]
    source: str


@dataclass
class LocalTrackRecord:
    """One output record for a local tracked detection."""

    scene_id: int
    scene_name: str
    split: str
    camera_id: str
    frame_id: int
    local_track_id: int
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
    track_age: int
    track_hits: int
    track_misses: int
    track_state: str


def detection_from_observation(obs: Observation3D) -> LocalTrackDetection:
    """Convert an Observation3D object to a LocalTrackDetection."""
    return LocalTrackDetection(
        scene_id=int(obs.scene_id),
        scene_name=str(obs.scene_name),
        split=str(obs.split),
        camera_id=str(obs.camera_id),
        frame_id=int(obs.frame_id),
        detection_id=int(obs.detection_id),
        class_id=int(obs.class_id),
        class_name=str(obs.class_name),
        confidence=float(obs.confidence),
        bbox_xyxy=tuple(float(value) for value in obs.bbox_xyxy),
        bbox_xywh=tuple(float(value) for value in obs.bbox_xywh),
        center_3d=_copy_array(obs.center_3d),
        dimensions_3d=_copy_array(obs.dimensions_3d),
        yaw=None if obs.yaw is None else float(obs.yaw),
        object_id=None if obs.object_id is None else int(obs.object_id),
        matched_gt=bool(obs.matched_gt),
        matched_iou=None if obs.matched_iou is None else float(obs.matched_iou),
        source=str(obs.source),
    )


def record_from_track_detection(track: "LocalTrackLike", det: LocalTrackDetection) -> LocalTrackRecord:
    """Create a LocalTrackRecord from a track-like object and detection."""
    return LocalTrackRecord(
        scene_id=det.scene_id,
        scene_name=det.scene_name,
        split=det.split,
        camera_id=det.camera_id,
        frame_id=det.frame_id,
        local_track_id=int(track.local_track_id),
        detection_id=det.detection_id,
        class_id=det.class_id,
        class_name=det.class_name,
        confidence=det.confidence,
        bbox_xyxy=det.bbox_xyxy,
        bbox_xywh=det.bbox_xywh,
        center_3d=_copy_array(det.center_3d),
        dimensions_3d=_copy_array(det.dimensions_3d),
        yaw=det.yaw,
        matched_gt_object_id=det.object_id,
        matched_gt=det.matched_gt,
        track_age=int(track.age),
        track_hits=int(track.hits),
        track_misses=int(track.misses),
        track_state=str(track.state),
    )


class LocalTrackLike:
    """Protocol-like small class for type checking without typing.Protocol."""

    local_track_id: int
    age: int
    hits: int
    misses: int
    state: str


def array_to_list(value: Optional[np.ndarray]) -> Optional[List[float]]:
    """Convert an optional numpy array to a JSON-friendly list."""
    if value is None:
        return None
    return [float(item) for item in np.asarray(value, dtype=float).reshape(-1)]


def list_to_array(value: object) -> Optional[np.ndarray]:
    """Convert an optional list-like value to numpy array."""
    if value is None or value == "":
        return None
    return np.asarray(value, dtype=float)


def _copy_array(value: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if value is None:
        return None
    return np.asarray(value, dtype=float).copy()
