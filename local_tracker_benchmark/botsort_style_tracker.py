"""Internal BoT-SORT-style tracker with optional Person appearance matching."""

from typing import Any, Dict, List, Optional

from deep_oc_sort_3d.local_tracker_benchmark.bytetrack_style_tracker import (
    ByteTrackStyleTracker,
    mark_missed,
    match_tracks,
    record_from_state,
    update_track,
)
from deep_oc_sort_3d.local_tracker_benchmark.tracker_input_types import BenchmarkDetection, BenchmarkTrackRecord


class BoTSORTStyleTracker(ByteTrackStyleTracker):
    """Motion-predicted ByteTrack stages plus optional Person appearance cost."""

    def __init__(self, config: Optional[Dict[str, Any]] = None, use_reid: bool = False) -> None:
        super(BoTSORTStyleTracker, self).__init__(config)
        settings = config or {}
        self.use_reid = bool(use_reid)
        self.appearance_weight = float(settings.get("appearance_weight", 0.35))
        self.appearance_thresh = float(settings.get("appearance_thresh", 0.25))

    def update(self, frame_id: int, detections: List[BenchmarkDetection]) -> List[BenchmarkTrackRecord]:
        high = [det for det in detections if det.confidence >= self.high_thresh]
        low = [det for det in detections if self.low_thresh <= det.confidence < self.high_thresh]
        for track in self.tracks:
            if int(frame_id) - int(track.last_frame) > self.track_buffer:
                track.state = "dead"
        active = [
            track for track in self.tracks
            if track.state != "dead" and track.misses <= self.track_buffer
        ]
        first_matches, unmatched_tracks, unmatched_high = match_tracks(
            active,
            high,
            frame_id,
            self.match_thresh,
            use_prediction=True,
            appearance_weight=self.appearance_weight if self.use_reid else 0.0,
            appearance_thresh=self.appearance_thresh if self.use_reid else None,
        )
        records = []
        matched_ids = set()
        for track_index, detection_index in first_matches:
            track = active[track_index]
            detection = high[detection_index]
            update_track(track, detection)
            records.append(record_from_state(track, detection))
            matched_ids.add(track.track_id)
        remaining = [active[index] for index in unmatched_tracks]
        second_matches, unmatched_second, _unmatched_low = match_tracks(
            remaining, low, frame_id, self.second_match_thresh, use_prediction=True
        )
        for track_index, detection_index in second_matches:
            track = remaining[track_index]
            detection = low[detection_index]
            update_track(track, detection)
            records.append(record_from_state(track, detection))
            matched_ids.add(track.track_id)
        unmatched_ids = set([remaining[index].track_id for index in unmatched_second])
        for track in active:
            if track.track_id not in matched_ids and track.track_id in unmatched_ids:
                mark_missed(track, frame_id, self.track_buffer)
        for detection_index in unmatched_high:
            detection = high[detection_index]
            if detection.confidence >= self.new_track_thresh:
                track = self._new_track(detection)
                records.append(record_from_state(track, detection))
        self.tracks = [track for track in self.tracks if track.state != "dead"]
        return sorted(records, key=lambda item: (item.frame_id, item.track_id))
