"""Local track state and lifecycle management."""

from typing import Dict, Optional

from deep_oc_sort_3d.tracking.motion_model import ConstantVelocity2D, ConstantVelocity3D
from deep_oc_sort_3d.tracking.track_types import LocalTrackDetection


class LocalTrack:
    """One local single-camera track."""

    def __init__(self, local_track_id: int, det: LocalTrackDetection):
        self.local_track_id = int(local_track_id)
        self.class_id = int(det.class_id)
        self.class_name = str(det.class_name)
        self.state = "tentative"
        self.age = 1
        self.hits = 1
        self.misses = 0
        self.first_frame = int(det.frame_id)
        self.last_frame = int(det.frame_id)
        self.last_detection = det
        self.motion_3d = ConstantVelocity3D()
        self.motion_2d = ConstantVelocity2D()
        self.gt_object_history = {}
        self._update_motion(det)
        self._update_gt_history(det)

    def predict(self, frame_id: int) -> Dict[str, object]:
        """Predict current 3D and 2D positions."""
        return {
            "center_3d": self.motion_3d.predict(frame_id),
            "bbox_center_2d": self.motion_2d.predict(frame_id),
        }

    def update(self, det: LocalTrackDetection) -> None:
        """Update this track with a matched detection."""
        self.last_detection = det
        self.last_frame = int(det.frame_id)
        self.age = max(self.age + 1, int(self.last_frame) - int(self.first_frame) + 1)
        self.hits += 1
        self.misses = 0
        if self.state in ("lost", "dead"):
            self.state = "confirmed"
        self._update_motion(det)
        self._update_gt_history(det)

    def mark_missed(self) -> None:
        """Mark one missed frame."""
        self.age += 1
        self.misses += 1
        if self.state != "dead":
            self.state = "lost"

    def is_confirmed(self, min_hits: int) -> bool:
        """Return True when the track has enough hits."""
        return self.hits >= int(min_hits)

    def is_dead(self, max_misses: int) -> bool:
        """Return True when the track exceeded the miss budget."""
        if self.misses > int(max_misses):
            self.state = "dead"
            return True
        return False

    def majority_gt_object_id(self) -> Optional[int]:
        """Return the most frequent matched GT object id for diagnostics."""
        if not self.gt_object_history:
            return None
        return max(self.gt_object_history.items(), key=lambda item: item[1])[0]

    def _update_motion(self, det: LocalTrackDetection) -> None:
        if det.center_3d is not None:
            self.motion_3d.update(det.center_3d, det.frame_id)
        self.motion_2d.update(det.bbox_xyxy, det.frame_id)

    def _update_gt_history(self, det: LocalTrackDetection) -> None:
        if det.object_id is None:
            return
        object_id = int(det.object_id)
        self.gt_object_history[object_id] = self.gt_object_history.get(object_id, 0) + 1
