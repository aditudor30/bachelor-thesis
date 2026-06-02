"""Build LocalTracklet objects from frame-level LocalTrackRecord rows."""

from typing import Dict, List, Optional, Tuple

import numpy as np

from deep_oc_sort_3d.tracking.track_types import LocalTrackRecord
from deep_oc_sort_3d.tracklets.tracklet_filtering import (
    classify_tracklet_quality,
    compute_tracklet_quality_score,
)
from deep_oc_sort_3d.tracklets.tracklet_types import LocalTracklet


class LocalTrackletBuilder:
    """Build local camera-level tracklets from local tracking records."""

    def __init__(
        self,
        min_length: int = 3,
        min_mean_confidence: float = 0.01,
        max_gap_allowed: int = 45,
        smooth_trajectory: bool = True,
        smoothing_window: int = 5,
    ):
        self.min_length = int(min_length)
        self.min_mean_confidence = float(min_mean_confidence)
        self.max_gap_allowed = int(max_gap_allowed)
        self.smooth_trajectory = bool(smooth_trajectory)
        self.smoothing_window = int(max(int(smoothing_window), 1))

    def build_from_records(self, records: List[LocalTrackRecord]) -> List[LocalTracklet]:
        """Build tracklets for all local tracks in records."""
        grouped = self.group_records_by_track(records)
        tracklets = []
        for _track_id, items in sorted(grouped.items(), key=lambda item: item[0]):
            tracklets.append(self.build_one_tracklet(items))
        return tracklets

    def build_one_tracklet(self, track_records: List[LocalTrackRecord]) -> LocalTracklet:
        """Build one LocalTracklet from records belonging to one local track."""
        records = sorted(track_records, key=lambda record: (record.frame_id, record.detection_id))
        if not records:
            raise ValueError("Cannot build a tracklet from zero records.")
        notes = []
        first = records[0]
        frame_ids = [int(record.frame_id) for record in records]
        detection_ids = [int(record.detection_id) for record in records]
        confidences = np.asarray([float(record.confidence) for record in records], dtype=float)

        class_id, class_name, class_note = _majority_class(records)
        if class_note:
            notes.append(class_note)

        bboxes = np.asarray([record.bbox_xyxy for record in records], dtype=float)
        if self.smooth_trajectory:
            bboxes_for_traj = _smooth_matrix(bboxes, self.smoothing_window)
        else:
            bboxes_for_traj = bboxes
        trajectory_2d = _trajectory_2d(frame_ids, bboxes_for_traj)

        centers, center_frames = _valid_arrays(records, "center_3d", 3)
        if centers.size > 0 and self.smooth_trajectory:
            centers_for_traj = _smooth_matrix(centers, self.smoothing_window)
        else:
            centers_for_traj = centers
        trajectory_3d = _trajectory_3d(center_frames, centers_for_traj)
        if len(trajectory_3d) < len(records):
            notes.append("missing center_3d for %d records" % (len(records) - len(trajectory_3d)))

        dimensions, _dimension_frames = _valid_arrays(records, "dimensions_3d", 3)
        yaws = [float(record.yaw) for record in records if record.yaw is not None]
        gt_counts = _gt_counts(records)
        majority_gt, gt_purity = _gt_majority(gt_counts)

        gaps = _frame_gaps(frame_ids)
        if gaps and max(gaps) > self.max_gap_allowed:
            notes.append("max frame gap %d exceeds max_gap_allowed %d" % (max(gaps), self.max_gap_allowed))

        tracklet = LocalTracklet(
            scene_id=int(first.scene_id),
            scene_name=str(first.scene_name),
            split=str(first.split),
            camera_id=str(first.camera_id),
            local_track_id=int(first.local_track_id),
            class_id=int(class_id),
            class_name=str(class_name),
            start_frame=int(min(frame_ids)),
            end_frame=int(max(frame_ids)),
            length=len(records),
            frame_ids=frame_ids,
            detection_ids=detection_ids,
            mean_confidence=float(np.mean(confidences)),
            median_confidence=float(np.median(confidences)),
            max_confidence=float(np.max(confidences)),
            bbox_start=_tuple4(bboxes_for_traj[0]),
            bbox_end=_tuple4(bboxes_for_traj[-1]),
            bbox_mean=_tuple4(np.mean(bboxes_for_traj, axis=0)),
            center_3d_start=_row_or_none(centers_for_traj, 0),
            center_3d_end=_row_or_none(centers_for_traj, -1),
            center_3d_mean=_mean_or_none(centers),
            center_3d_median=_median_or_none(centers),
            dimensions_3d_mean=_mean_or_none(dimensions),
            yaw_mean=None if not yaws else float(np.mean(np.asarray(yaws, dtype=float))),
            trajectory_2d=trajectory_2d,
            trajectory_3d=trajectory_3d,
            majority_gt_object_id=majority_gt,
            gt_purity=gt_purity,
            num_gt_ids=len(gt_counts),
            gt_id_counts={str(key): int(value) for key, value in gt_counts.items()},
            quality_score=0.0,
            quality_flag="invalid",
            is_valid_for_mtmc=False,
            notes="; ".join(notes),
        )
        tracklet.quality_score = compute_tracklet_quality_score(
            tracklet,
            min_length=self.min_length,
            min_mean_confidence=self.min_mean_confidence,
            prefer_3d=True,
        )
        flag, is_valid, merged_notes = classify_tracklet_quality(
            tracklet,
            min_length=self.min_length,
            min_mean_confidence=self.min_mean_confidence,
            min_gt_purity=None,
        )
        tracklet.quality_flag = flag
        tracklet.is_valid_for_mtmc = is_valid
        tracklet.notes = merged_notes
        return tracklet

    def group_records_by_track(self, records: List[LocalTrackRecord]) -> Dict[int, List[LocalTrackRecord]]:
        """Group records by local_track_id."""
        grouped = {}
        for record in records:
            key = int(record.local_track_id)
            if key not in grouped:
                grouped[key] = []
            grouped[key].append(record)
        return grouped


def _majority_class(records: List[LocalTrackRecord]) -> Tuple[int, str, str]:
    counts = {}
    names = {}
    for record in records:
        class_id = int(record.class_id)
        counts[class_id] = counts.get(class_id, 0) + 1
        names[class_id] = str(record.class_name)
    class_id = max(counts.items(), key=lambda item: item[1])[0]
    note = ""
    if len(counts) > 1:
        note = "multiple class ids in local track: %s" % counts
    return class_id, names.get(class_id, ""), note


def _valid_arrays(records: List[LocalTrackRecord], field: str, size: int) -> Tuple[np.ndarray, List[int]]:
    rows = []
    frames = []
    for record in records:
        value = getattr(record, field)
        if value is None:
            continue
        arr = np.asarray(value, dtype=float).reshape(-1)
        if arr.size < size:
            continue
        rows.append(arr[:size])
        frames.append(int(record.frame_id))
    if not rows:
        return np.zeros((0, size), dtype=float), []
    return np.vstack(rows), frames


def _smooth_matrix(values: np.ndarray, window: int) -> np.ndarray:
    if values.size == 0 or int(window) <= 1 or values.shape[0] <= 1:
        return values.copy()
    radius = int(window) // 2
    output = []
    for index in range(values.shape[0]):
        start = max(0, index - radius)
        end = min(values.shape[0], index + radius + 1)
        output.append(np.mean(values[start:end], axis=0))
    return np.asarray(output, dtype=float)


def _trajectory_2d(frame_ids: List[int], bboxes: np.ndarray) -> List[Tuple[int, float, float, float, float]]:
    output = []
    for frame_id, bbox in zip(frame_ids, bboxes):
        output.append((int(frame_id), float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])))
    return output


def _trajectory_3d(frame_ids: List[int], centers: np.ndarray) -> List[Tuple[int, float, float, float]]:
    output = []
    for frame_id, center in zip(frame_ids, centers):
        output.append((int(frame_id), float(center[0]), float(center[1]), float(center[2])))
    return output


def _tuple4(value: np.ndarray) -> Tuple[float, float, float, float]:
    arr = np.asarray(value, dtype=float).reshape(-1)
    return (float(arr[0]), float(arr[1]), float(arr[2]), float(arr[3]))


def _row_or_none(values: np.ndarray, index: int) -> Optional[np.ndarray]:
    if values.size == 0:
        return None
    return np.asarray(values[index], dtype=float).copy()


def _mean_or_none(values: np.ndarray) -> Optional[np.ndarray]:
    if values.size == 0:
        return None
    return np.mean(values, axis=0)


def _median_or_none(values: np.ndarray) -> Optional[np.ndarray]:
    if values.size == 0:
        return None
    return np.median(values, axis=0)


def _gt_counts(records: List[LocalTrackRecord]) -> Dict[int, int]:
    counts = {}
    for record in records:
        if record.matched_gt_object_id is None:
            continue
        object_id = int(record.matched_gt_object_id)
        counts[object_id] = counts.get(object_id, 0) + 1
    return counts


def _gt_majority(counts: Dict[int, int]) -> Tuple[Optional[int], Optional[float]]:
    if not counts:
        return None, None
    total = sum(counts.values())
    object_id, count = max(counts.items(), key=lambda item: item[1])
    return int(object_id), float(count) / float(total)


def _frame_gaps(frame_ids: List[int]) -> List[int]:
    frames = sorted(set(int(item) for item in frame_ids))
    return [int(frames[index]) - int(frames[index - 1]) for index in range(1, len(frames))]
