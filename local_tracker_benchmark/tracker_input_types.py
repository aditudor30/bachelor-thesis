"""Dataclasses shared by local tracker benchmark variants."""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np


@dataclass
class BenchmarkDetection:
    """One YOLO detection with optional diagnostic identity and appearance."""

    scene_id: int
    scene_name: str
    subset: str
    split: str
    camera_id: str
    frame_id: int
    detection_id: int
    class_id: int
    class_name: str
    confidence: float
    bbox_xyxy: Tuple[float, float, float, float]
    matched_gt_object_id: Optional[int] = None
    embedding: Optional[np.ndarray] = None


@dataclass
class BenchmarkTrackRecord:
    """One benchmark-local tracked detection."""

    scene_id: int
    scene_name: str
    subset: str
    split: str
    camera_id: str
    frame_id: int
    track_id: int
    detection_id: int
    class_id: int
    class_name: str
    confidence: float
    bbox_xyxy: Tuple[float, float, float, float]
    matched_gt_object_id: Optional[int]
    track_age: int
    track_hits: int
    track_misses: int
    track_state: str
    source_detection_id: int


@dataclass
class BenchmarkTrackState:
    """Mutable state for internal ByteTrack/BoT-SORT-style variants."""

    track_id: int
    class_id: int
    class_name: str
    bbox_xyxy: Tuple[float, float, float, float]
    last_frame: int
    first_frame: int
    confidence: float
    hits: int = 1
    misses: int = 0
    state: str = "tentative"
    velocity_xy: Tuple[float, float] = (0.0, 0.0)
    embedding: Optional[np.ndarray] = None
    history: List[Tuple[int, Tuple[float, float, float, float]]] = field(default_factory=list)

    def predicted_bbox(self, frame_id: int) -> Tuple[float, float, float, float]:
        """Return a constant-velocity bbox prediction."""
        gap = max(0, int(frame_id) - int(self.last_frame))
        dx = self.velocity_xy[0] * float(gap)
        dy = self.velocity_xy[1] * float(gap)
        x1, y1, x2, y2 = self.bbox_xyxy
        return (x1 + dx, y1 + dy, x2 + dx, y2 + dy)
