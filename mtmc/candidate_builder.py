"""Build MTMCTrackletCandidate objects from LocalTracklet objects."""

from typing import Iterable, List, Optional, Tuple

import numpy as np

from deep_oc_sort_3d.mtmc.candidate_filtering import (
    compute_candidate_quality_score,
    should_keep_tracklet,
)
from deep_oc_sort_3d.mtmc.candidate_types import MTMCTrackletCandidate, make_candidate_id
from deep_oc_sort_3d.tracklets.tracklet_types import LocalTracklet


class MTMCCandidateBuilder:
    """Convert LocalTracklet objects into compact MTMC candidate records."""

    def __init__(
        self,
        min_length: int = 3,
        min_mean_confidence: float = 0.01,
        allowed_quality_flags: Optional[List[str]] = None,
        require_valid_for_mtmc: bool = True,
        require_3d: bool = False,
        trajectory_sample_rate: int = 5,
        max_trajectory_points: int = 50,
        class_allowlist: Optional[List[str]] = None,
        class_blocklist: Optional[List[str]] = None,
    ):
        self.min_length = int(min_length)
        self.min_mean_confidence = float(min_mean_confidence)
        self.allowed_quality_flags = allowed_quality_flags
        self.require_valid_for_mtmc = bool(require_valid_for_mtmc)
        self.require_3d = bool(require_3d)
        self.trajectory_sample_rate = max(int(trajectory_sample_rate), 1)
        self.max_trajectory_points = max(int(max_trajectory_points), 2)
        self.class_allowlist = class_allowlist
        self.class_blocklist = class_blocklist

    def build_from_tracklet(self, tracklet: LocalTracklet, subset: str) -> MTMCTrackletCandidate:
        """Build one candidate from one LocalTracklet."""
        keep, reject_reason = should_keep_tracklet(
            tracklet=tracklet,
            min_length=self.min_length,
            min_mean_confidence=self.min_mean_confidence,
            allowed_quality_flags=self.allowed_quality_flags,
            require_valid_for_mtmc=self.require_valid_for_mtmc,
            require_3d=self.require_3d,
            class_allowlist=self.class_allowlist,
            class_blocklist=self.class_blocklist,
        )
        sampled_2d = sample_trajectory_2d(
            tracklet.trajectory_2d,
            sample_rate=self.trajectory_sample_rate,
            max_points=self.max_trajectory_points,
        )
        sampled_3d = sample_trajectory_3d(
            tracklet.trajectory_3d,
            sample_rate=self.trajectory_sample_rate,
            max_points=self.max_trajectory_points,
        )
        entry_frame, exit_frame, entry_center, exit_center = _entry_exit_3d(tracklet)
        mean_velocity = _mean_velocity_3d(tracklet.trajectory_3d)
        travel_distance = _travel_distance_3d(tracklet.trajectory_3d)
        duration = int(tracklet.end_frame) - int(tracklet.start_frame) + 1
        return MTMCTrackletCandidate(
            candidate_id=make_candidate_id(tracklet.scene_name, tracklet.camera_id, tracklet.local_track_id),
            scene_id=int(tracklet.scene_id),
            scene_name=str(tracklet.scene_name),
            split=str(tracklet.split),
            subset=str(subset),
            camera_id=str(tracklet.camera_id),
            local_track_id=int(tracklet.local_track_id),
            class_id=int(tracklet.class_id),
            class_name=str(tracklet.class_name),
            start_frame=int(tracklet.start_frame),
            end_frame=int(tracklet.end_frame),
            length=int(tracklet.length),
            duration=int(duration),
            mean_confidence=float(tracklet.mean_confidence),
            median_confidence=float(tracklet.median_confidence),
            max_confidence=float(tracklet.max_confidence),
            quality_score=compute_candidate_quality_score(tracklet),
            quality_flag=str(tracklet.quality_flag),
            source_tracklet_valid_for_mtmc=bool(tracklet.is_valid_for_mtmc),
            is_candidate=bool(keep),
            reject_reason=reject_reason,
            bbox_start=tracklet.bbox_start,
            bbox_end=tracklet.bbox_end,
            bbox_mean=tracklet.bbox_mean,
            center_3d_start=_copy_array(tracklet.center_3d_start),
            center_3d_end=_copy_array(tracklet.center_3d_end),
            center_3d_mean=_copy_array(tracklet.center_3d_mean),
            center_3d_median=_copy_array(tracklet.center_3d_median),
            trajectory_2d_sampled=sampled_2d,
            trajectory_3d_sampled=sampled_3d,
            trajectory_3d_length=len(tracklet.trajectory_3d),
            has_3d=bool(tracklet.trajectory_3d),
            entry_frame=entry_frame,
            exit_frame=exit_frame,
            entry_center_3d=entry_center,
            exit_center_3d=exit_center,
            mean_velocity_3d=mean_velocity,
            travel_distance_3d=travel_distance,
            majority_gt_object_id=tracklet.majority_gt_object_id,
            gt_purity=tracklet.gt_purity,
            num_gt_ids=int(tracklet.num_gt_ids),
            gt_id_counts=dict(tracklet.gt_id_counts),
            reid_embedding_path=None,
            reid_embedding=None,
            global_track_id=None,
        )

    def build_from_tracklets(
        self,
        tracklets: List[LocalTracklet],
        subset: str,
        show_progress: bool = True,
    ) -> List[MTMCTrackletCandidate]:
        """Build candidates for a list of tracklets."""
        output = []
        for tracklet in _progress_iter(tracklets, show_progress, "build MTMC candidates"):
            output.append(self.build_from_tracklet(tracklet, subset=subset))
        return output


def sample_trajectory_2d(
    trajectory: List[Tuple[int, float, float, float, float]],
    sample_rate: int = 5,
    max_points: int = 50,
) -> List[Tuple[int, float, float, float, float]]:
    """Sample 2D trajectory while preserving endpoints."""
    sampled = _sample_sequence(trajectory, sample_rate, max_points)
    return [(int(a), float(b), float(c), float(d), float(e)) for a, b, c, d, e in sampled]


def sample_trajectory_3d(
    trajectory: List[Tuple[int, float, float, float]],
    sample_rate: int = 5,
    max_points: int = 50,
) -> List[Tuple[int, float, float, float]]:
    """Sample 3D trajectory while preserving endpoints."""
    sampled = _sample_sequence(trajectory, sample_rate, max_points)
    return [(int(a), float(b), float(c), float(d)) for a, b, c, d in sampled]


def _sample_sequence(items: List[Tuple], sample_rate: int, max_points: int) -> List[Tuple]:
    if not items:
        return []
    if len(items) <= max_points:
        sampled = items[:: max(int(sample_rate), 1)]
        if items[0] not in sampled:
            sampled.insert(0, items[0])
        if items[-1] not in sampled:
            sampled.append(items[-1])
        return sampled
    indices = np.linspace(0, len(items) - 1, num=max_points, dtype=int)
    indices = sorted(set(int(index) for index in indices))
    if 0 not in indices:
        indices.insert(0, 0)
    if len(items) - 1 not in indices:
        indices.append(len(items) - 1)
    return [items[index] for index in indices[:max_points]]


def _entry_exit_3d(tracklet: LocalTracklet) -> Tuple[int, int, Optional[np.ndarray], Optional[np.ndarray]]:
    if not tracklet.trajectory_3d:
        return int(tracklet.start_frame), int(tracklet.end_frame), None, None
    first = tracklet.trajectory_3d[0]
    last = tracklet.trajectory_3d[-1]
    return (
        int(first[0]),
        int(last[0]),
        np.asarray([first[1], first[2], first[3]], dtype=float),
        np.asarray([last[1], last[2], last[3]], dtype=float),
    )


def _mean_velocity_3d(trajectory: List[Tuple[int, float, float, float]]) -> Optional[np.ndarray]:
    if len(trajectory) < 2:
        return None
    first = trajectory[0]
    last = trajectory[-1]
    dt = int(last[0]) - int(first[0])
    if dt <= 0:
        return None
    start = np.asarray([first[1], first[2], first[3]], dtype=float)
    end = np.asarray([last[1], last[2], last[3]], dtype=float)
    return (end - start) / float(dt)


def _travel_distance_3d(trajectory: List[Tuple[int, float, float, float]]) -> Optional[float]:
    if len(trajectory) < 2:
        return None
    distance = 0.0
    previous = np.asarray([trajectory[0][1], trajectory[0][2], trajectory[0][3]], dtype=float)
    for item in trajectory[1:]:
        current = np.asarray([item[1], item[2], item[3]], dtype=float)
        distance += float(np.linalg.norm(current - previous))
        previous = current
    return float(distance)


def _copy_array(value: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if value is None:
        return None
    return np.asarray(value, dtype=float).copy()


def _progress_iter(values: List[LocalTracklet], show_progress: bool, desc: str) -> Iterable[LocalTracklet]:
    if not show_progress:
        return values
    try:
        from tqdm import tqdm
    except ImportError:
        return _print_progress_iter(values, desc)
    return tqdm(values, desc=desc, unit="tracklet")


def _print_progress_iter(values: List[LocalTracklet], desc: str) -> Iterable[LocalTracklet]:
    total = len(values)
    for index, value in enumerate(values):
        if index == 0 or (index + 1) % 1000 == 0 or index + 1 == total:
            print("%s: tracklet %d/%d" % (desc, index + 1, total))
        yield value
