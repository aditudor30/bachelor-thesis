"""Prototype local single-camera tracker over Observation3D records."""

from typing import Any, Dict, Iterable, List, Optional

from deep_oc_sort_3d.observations.observation_types import Observation3D
from deep_oc_sort_3d.tracking.association import associate_detections_to_tracks
from deep_oc_sort_3d.tracking.track_state import LocalTrack
from deep_oc_sort_3d.tracking.track_types import (
    LocalTrackDetection,
    LocalTrackRecord,
    detection_from_observation,
    record_from_track_detection,
)


class LocalObservationTracker:
    """Track local single-camera Observation3D detections."""

    def __init__(
        self,
        mode: str = "hybrid",
        min_confidence: float = 0.01,
        min_hits: int = 2,
        max_misses: int = 30,
        cost_threshold: float = 0.7,
        max_3d_distance: float = 3.0,
        min_iou: float = 0.05,
        class_must_match: bool = True,
        max_detections_per_frame: Optional[int] = None,
        per_class_conf_thresholds: Optional[Dict[str, float]] = None,
    ):
        self.mode = mode
        self.min_confidence = float(min_confidence)
        self.min_hits = int(min_hits)
        self.max_misses = int(max_misses)
        self.cost_threshold = float(cost_threshold)
        self.max_3d_distance = float(max_3d_distance)
        self.min_iou = float(min_iou)
        self.class_must_match = bool(class_must_match)
        self.max_detections_per_frame = max_detections_per_frame
        self.per_class_conf_thresholds = dict(per_class_conf_thresholds or {})
        self.tracks = []
        self.next_track_id = 1
        self.num_frames = 0
        self.num_input_detections = 0
        self.num_kept_detections = 0

    def update(self, frame_id: int, detections: List[LocalTrackDetection]) -> List[LocalTrackRecord]:
        """Update tracker for one frame."""
        self.num_frames += 1
        self.num_input_detections += len(detections)
        detections = self._filter_detections(detections)
        self.num_kept_detections += len(detections)

        active_tracks = [track for track in self.tracks if track.state != "dead"]
        config = {
            "mode": self.mode,
            "cost_threshold": self.cost_threshold,
            "max_3d_distance": self.max_3d_distance,
            "min_iou": self.min_iou,
            "class_must_match": self.class_must_match,
        }
        matched, unmatched_track_indices, unmatched_detection_indices = associate_detections_to_tracks(
            detections=detections,
            tracks=active_tracks,
            frame_id=frame_id,
            config=config,
        )

        records = []
        for track_index, detection_index in matched:
            track = active_tracks[track_index]
            det = detections[detection_index]
            track.update(det)
            if track.is_confirmed(self.min_hits):
                track.state = "confirmed"
            records.append(record_from_track_detection(track, det))

        for track_index in unmatched_track_indices:
            track = active_tracks[track_index]
            track.mark_missed()
            track.is_dead(self.max_misses)

        for detection_index in unmatched_detection_indices:
            det = detections[detection_index]
            track = LocalTrack(self.next_track_id, det)
            self.next_track_id += 1
            if track.is_confirmed(self.min_hits):
                track.state = "confirmed"
            self.tracks.append(track)
            records.append(record_from_track_detection(track, det))

        self.tracks = [track for track in self.tracks if track.state != "dead"]
        return sorted(records, key=lambda record: (record.frame_id, record.local_track_id, record.detection_id))

    def run(self, observations: List[Observation3D], show_progress: bool = True) -> List[LocalTrackRecord]:
        """Run tracker over a list of Observation3D records."""
        detections_by_frame = _group_observations_by_frame(observations)
        records = []
        frame_ids = sorted(detections_by_frame.keys())
        for frame_id in _progress_iter(frame_ids, show_progress, "local tracking"):
            records.extend(self.update(frame_id, detections_by_frame[frame_id]))
        return records

    def summary(self) -> Dict[str, Any]:
        """Return tracker-level summary."""
        active = [track for track in self.tracks if track.state != "dead"]
        return {
            "num_frames": self.num_frames,
            "num_input_detections": self.num_input_detections,
            "num_kept_detections": self.num_kept_detections,
            "num_active_tracks": len(active),
            "next_track_id": self.next_track_id,
        }

    def _filter_detections(self, detections: List[LocalTrackDetection]) -> List[LocalTrackDetection]:
        kept = []
        for det in detections:
            threshold = float(self.per_class_conf_thresholds.get(det.class_name, self.min_confidence))
            if det.confidence < threshold:
                continue
            kept.append(det)
        kept = sorted(kept, key=lambda item: item.confidence, reverse=True)
        if self.max_detections_per_frame is not None:
            kept = kept[: int(self.max_detections_per_frame)]
        return kept


def _group_observations_by_frame(observations: List[Observation3D]) -> Dict[int, List[LocalTrackDetection]]:
    grouped = {}
    for obs in observations:
        det = detection_from_observation(obs)
        if det.frame_id not in grouped:
            grouped[det.frame_id] = []
        grouped[det.frame_id].append(det)
    return grouped


def _progress_iter(values: Iterable[int], show_progress: bool, desc: str) -> Iterable[int]:
    items = list(values)
    if not show_progress:
        return items
    try:
        from tqdm import tqdm
    except ImportError:
        return _print_progress_iter(items, desc)
    return tqdm(items, desc=desc, unit="frame")


def _print_progress_iter(values: List[int], desc: str) -> Iterable[int]:
    total = len(values)
    for index, value in enumerate(values):
        if index == 0 or (index + 1) % 100 == 0 or index + 1 == total:
            print("%s: frame %d/%d" % (desc, index + 1, total))
        yield value
