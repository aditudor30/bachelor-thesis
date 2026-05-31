"""Dataclasses for frame-level 3D training targets."""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np


@dataclass
class TrainingObjectTarget:
    """One object target for a future trainable 3D MTMC model."""

    scene_id: int
    scene_name: str
    frame_id: int
    camera_id: str
    class_name: str
    class_id: int
    object_id: int
    bbox_xyxy: Optional[Tuple[float, float, float, float]]
    center_3d: np.ndarray
    dimensions_3d: np.ndarray
    rotation_3d: np.ndarray
    yaw: float
    depth_value: Optional[float]
    backprojected_center_3d: Optional[np.ndarray]
    backprojection_error: Optional[float]


@dataclass
class FrameTrainingTargets:
    """Targets for one frame and one camera."""

    scene_id: int
    scene_name: str
    frame_id: int
    camera_id: str
    targets: List[TrainingObjectTarget]

