"""Internal multi-class ByteTrack-style tracker over existing YOLO detections."""

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from deep_oc_sort_3d.local_tracker_benchmark.tracker_input_types import (
    BenchmarkDetection,
    BenchmarkTrackRecord,
    BenchmarkTrackState,
)
from deep_oc_sort_3d.tracking.association import bbox_iou_xyxy


class ByteTrackStyleTracker:
    """Two-stage confidence association with class-safe IoU and track buffer."""

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        settings = config or {}
        self.high_thresh = float(settings.get("track_high_thresh", 0.3))
        self.low_thresh = float(settings.get("track_low_thresh", 0.05))
        self.new_track_thresh = float(settings.get("new_track_thresh", 0.4))
        self.match_thresh = float(settings.get("match_thresh", 0.8))
        self.second_match_thresh = float(settings.get("second_stage_match_thresh", 0.5))
        self.track_buffer = int(settings.get("track_buffer", 45))
        self.tracks = []  # type: List[BenchmarkTrackState]
        self.next_track_id = 1

    def update(self, frame_id: int, detections: List[BenchmarkDetection]) -> List[BenchmarkTrackRecord]:
        """Update one frame using high detections first and low detections second."""
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
            active, high, frame_id, self.match_thresh, use_prediction=False
        )
        records = []
        matched_track_ids = set()
        for track_index, detection_index in first_matches:
            track = active[track_index]
            detection = high[detection_index]
            update_track(track, detection)
            records.append(record_from_state(track, detection))
            matched_track_ids.add(track.track_id)
        remaining_tracks = [active[index] for index in unmatched_tracks]
        second_matches, unmatched_second_tracks, _unmatched_low = match_tracks(
            remaining_tracks, low, frame_id, self.second_match_thresh, use_prediction=False
        )
        for track_index, detection_index in second_matches:
            track = remaining_tracks[track_index]
            detection = low[detection_index]
            update_track(track, detection)
            records.append(record_from_state(track, detection))
            matched_track_ids.add(track.track_id)
        unmatched_track_ids = set([remaining_tracks[index].track_id for index in unmatched_second_tracks])
        for track in active:
            if track.track_id not in matched_track_ids and track.track_id in unmatched_track_ids:
                mark_missed(track, frame_id, self.track_buffer)
        for detection_index in unmatched_high:
            detection = high[detection_index]
            if detection.confidence < self.new_track_thresh:
                continue
            track = self._new_track(detection)
            records.append(record_from_state(track, detection))
        self.tracks = [track for track in self.tracks if track.state != "dead"]
        return sorted(records, key=lambda item: (item.frame_id, item.track_id))

    def run(self, detections_by_frame: Dict[int, List[BenchmarkDetection]]) -> List[BenchmarkTrackRecord]:
        """Run all camera frames in temporal order."""
        records = []
        for frame_id in sorted(detections_by_frame.keys()):
            records.extend(self.update(frame_id, detections_by_frame[frame_id]))
        return records

    def _new_track(self, detection: BenchmarkDetection) -> BenchmarkTrackState:
        track = BenchmarkTrackState(
            track_id=self.next_track_id,
            class_id=detection.class_id,
            class_name=detection.class_name,
            bbox_xyxy=detection.bbox_xyxy,
            last_frame=detection.frame_id,
            first_frame=detection.frame_id,
            confidence=detection.confidence,
            embedding=_copy_embedding(detection.embedding),
            history=[(detection.frame_id, detection.bbox_xyxy)],
        )
        self.next_track_id += 1
        self.tracks.append(track)
        return track


def match_tracks(
    tracks: Sequence[BenchmarkTrackState],
    detections: Sequence[BenchmarkDetection],
    frame_id: int,
    match_thresh: float,
    use_prediction: bool,
    appearance_weight: float = 0.0,
    appearance_thresh: Optional[float] = None,
) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
    """Hungarian/greedy class-safe matching by IoU plus optional appearance."""
    if not tracks or not detections:
        return [], list(range(len(tracks))), list(range(len(detections)))
    cost = np.full((len(tracks), len(detections)), 1e9, dtype=np.float64)
    min_iou = max(0.0, 1.0 - float(match_thresh))
    for track_index, track in enumerate(tracks):
        box = track.predicted_bbox(frame_id) if use_prediction else track.bbox_xyxy
        for detection_index, detection in enumerate(detections):
            if int(track.class_id) != int(detection.class_id):
                continue
            iou = bbox_iou_xyxy(box, detection.bbox_xyxy)
            if iou < min_iou:
                continue
            value = 1.0 - iou
            if appearance_weight > 0.0 and track.class_id == 0:
                similarity = cosine_similarity(track.embedding, detection.embedding)
                if appearance_thresh is not None and similarity is not None and similarity < appearance_thresh:
                    continue
                if similarity is not None:
                    value = (1.0 - appearance_weight) * value + appearance_weight * (1.0 - similarity)
            cost[track_index, detection_index] = value
    matches = _solve_assignment(cost, max_cost=float(match_thresh))
    matched_tracks = set([left for left, _right in matches])
    matched_detections = set([right for _left, right in matches])
    return (
        matches,
        [index for index in range(len(tracks)) if index not in matched_tracks],
        [index for index in range(len(detections)) if index not in matched_detections],
    )


def update_track(track: BenchmarkTrackState, detection: BenchmarkDetection) -> None:
    """Update bbox velocity, appearance and lifecycle state."""
    old_center = bbox_center(track.bbox_xyxy)
    new_center = bbox_center(detection.bbox_xyxy)
    frame_delta = max(1, int(detection.frame_id) - int(track.last_frame))
    track.velocity_xy = (
        (new_center[0] - old_center[0]) / float(frame_delta),
        (new_center[1] - old_center[1]) / float(frame_delta),
    )
    track.bbox_xyxy = detection.bbox_xyxy
    track.last_frame = detection.frame_id
    track.confidence = detection.confidence
    track.hits += 1
    track.misses = 0
    track.state = "confirmed" if track.hits >= 2 else "tentative"
    track.history.append((detection.frame_id, detection.bbox_xyxy))
    if detection.embedding is not None:
        track.embedding = update_embedding(track.embedding, detection.embedding)


def mark_missed(track: BenchmarkTrackState, frame_id: int, track_buffer: int) -> None:
    """Age an unmatched track according to actual frame gap."""
    track.misses = max(track.misses + 1, int(frame_id) - int(track.last_frame))
    track.state = "dead" if track.misses > int(track_buffer) else "lost"


def record_from_state(track: BenchmarkTrackState, detection: BenchmarkDetection) -> BenchmarkTrackRecord:
    """Create one output record."""
    return BenchmarkTrackRecord(
        scene_id=detection.scene_id,
        scene_name=detection.scene_name,
        subset=detection.subset,
        split=detection.split,
        camera_id=detection.camera_id,
        frame_id=detection.frame_id,
        track_id=track.track_id,
        detection_id=detection.detection_id,
        class_id=detection.class_id,
        class_name=detection.class_name,
        confidence=detection.confidence,
        bbox_xyxy=detection.bbox_xyxy,
        matched_gt_object_id=detection.matched_gt_object_id,
        track_age=max(1, detection.frame_id - track.first_frame + 1),
        track_hits=track.hits,
        track_misses=track.misses,
        track_state=track.state,
        source_detection_id=detection.detection_id,
    )


def bbox_center(box: Tuple[float, float, float, float]) -> Tuple[float, float]:
    return ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)


def cosine_similarity(left: Any, right: Any) -> Optional[float]:
    if left is None or right is None:
        return None
    a = np.asarray(left, dtype=np.float64).reshape(-1)
    b = np.asarray(right, dtype=np.float64).reshape(-1)
    if a.size == 0 or a.shape != b.shape:
        return None
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    return None if denominator <= 1e-12 else float(np.dot(a, b) / denominator)


def update_embedding(current: Any, new_value: Any, momentum: float = 0.9) -> np.ndarray:
    new_array = np.asarray(new_value, dtype=np.float64).reshape(-1)
    if current is None:
        result = new_array
    else:
        result = momentum * np.asarray(current, dtype=np.float64).reshape(-1) + (1.0 - momentum) * new_array
    norm = float(np.linalg.norm(result))
    return result if norm <= 1e-12 else result / norm


def _solve_assignment(cost: np.ndarray, max_cost: float) -> List[Tuple[int, int]]:
    try:
        from scipy.optimize import linear_sum_assignment

        row_indices, column_indices = linear_sum_assignment(cost)
        candidates = [(int(row), int(column)) for row, column in zip(row_indices, column_indices)]
    except ImportError:
        candidates = []
        for row in range(cost.shape[0]):
            for column in range(cost.shape[1]):
                candidates.append((row, column))
        candidates.sort(key=lambda item: float(cost[item[0], item[1]]))
    output = []
    used_rows = set()
    used_columns = set()
    for row, column in candidates:
        value = float(cost[row, column])
        if value > max_cost or value >= 1e9 or row in used_rows or column in used_columns:
            continue
        output.append((row, column))
        used_rows.add(row)
        used_columns.add(column)
    return output


def _copy_embedding(value: Any) -> Optional[np.ndarray]:
    return None if value is None else np.asarray(value, dtype=np.float64).copy()
