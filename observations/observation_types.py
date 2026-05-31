"""Dataclasses for standardized 3D observations."""

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass
class Observation3D:
    """One standardized 3D observation created from a YOLO detection."""

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
    depth_value: Optional[float]
    depth_sampling_method: Optional[str]
    source: str


@dataclass
class MatchedDetectionGT:
    """A matched YOLO detection and visible GT object."""

    detection_id: int
    object_id: int
    iou: float
    class_id: int
    class_name: str
    frame_id: int
    camera_id: str

