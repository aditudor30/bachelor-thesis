"""Simple constant-velocity motion models for local tracking."""

from typing import Optional, Tuple

import numpy as np


class ConstantVelocity3D:
    """Constant-velocity model over 3D centers."""

    def __init__(self) -> None:
        self.last_center = None
        self.velocity = None
        self.last_frame = None

    def predict(self, frame_id: int) -> Optional[np.ndarray]:
        """Predict center at frame_id, returning the last center if velocity is unknown."""
        if self.last_center is None:
            return None
        if self.velocity is None or self.last_frame is None:
            return self.last_center.copy()
        gap = max(int(frame_id) - int(self.last_frame), 0)
        return self.last_center + self.velocity * float(gap)

    def update(self, center: np.ndarray, frame_id: int) -> None:
        """Update model state from a new center."""
        new_center = np.asarray(center, dtype=float)
        if self.last_center is not None and self.last_frame is not None:
            gap = int(frame_id) - int(self.last_frame)
            if gap > 0:
                self.velocity = (new_center - self.last_center) / float(gap)
        self.last_center = new_center.copy()
        self.last_frame = int(frame_id)


class ConstantVelocity2D:
    """Constant-velocity model over 2D bbox centers."""

    def __init__(self) -> None:
        self.last_bbox_center = None
        self.velocity = None
        self.last_frame = None

    def predict(self, frame_id: int) -> Optional[np.ndarray]:
        """Predict bbox center at frame_id, returning the last center if velocity is unknown."""
        if self.last_bbox_center is None:
            return None
        if self.velocity is None or self.last_frame is None:
            return self.last_bbox_center.copy()
        gap = max(int(frame_id) - int(self.last_frame), 0)
        return self.last_bbox_center + self.velocity * float(gap)

    def update(self, bbox_xyxy: Tuple[float, float, float, float], frame_id: int) -> None:
        """Update model state from a new bbox."""
        new_center = bbox_center_xyxy(bbox_xyxy)
        if self.last_bbox_center is not None and self.last_frame is not None:
            gap = int(frame_id) - int(self.last_frame)
            if gap > 0:
                self.velocity = (new_center - self.last_bbox_center) / float(gap)
        self.last_bbox_center = new_center.copy()
        self.last_frame = int(frame_id)


def bbox_center_xyxy(bbox_xyxy: Tuple[float, float, float, float]) -> np.ndarray:
    """Return the center of an xyxy bbox as [x, y]."""
    x1, y1, x2, y2 = bbox_xyxy
    return np.asarray([(float(x1) + float(x2)) * 0.5, (float(y1) + float(y2)) * 0.5], dtype=float)
