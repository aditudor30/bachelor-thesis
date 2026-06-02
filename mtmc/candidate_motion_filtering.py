"""Filtering helpers for candidate motion quality."""

from typing import Any, Dict, Iterable, List, Tuple

from deep_oc_sort_3d.mtmc.candidate_motion_quality import (
    CandidateMotionMetrics,
    compute_candidate_motion_metrics,
)
from deep_oc_sort_3d.mtmc.candidate_types import MTMCTrackletCandidate, candidate_to_dict


def attach_motion_metrics_to_candidate_dict(
    candidate: MTMCTrackletCandidate,
    metrics: CandidateMotionMetrics,
) -> Dict[str, Any]:
    """Return a candidate dictionary augmented with motion metrics."""
    data = candidate_to_dict(candidate)
    data.update(_metrics_to_compact_dict(metrics))
    return data


def split_candidates_by_motion_quality(
    candidates: List[MTMCTrackletCandidate],
    config: Dict[str, Any],
    show_progress: bool = True,
) -> Dict[str, List[MTMCTrackletCandidate]]:
    """Split candidates into clean/suspicious/invalid/unknown buckets."""
    buckets = {"clean": [], "suspicious": [], "invalid": [], "unknown": []}
    for candidate in _progress_iter(candidates, show_progress, "split motion quality"):
        metrics = compute_candidate_motion_metrics(candidate, config)
        if metrics.is_motion_clean:
            buckets["clean"].append(candidate)
        elif metrics.motion_quality_flag == "motion_suspicious":
            buckets["suspicious"].append(candidate)
        elif metrics.motion_quality_flag == "motion_unknown":
            buckets["unknown"].append(candidate)
        else:
            buckets["invalid"].append(candidate)
    return buckets


def filter_motion_clean_candidates(
    candidates: List[MTMCTrackletCandidate],
    config: Dict[str, Any],
    show_progress: bool = True,
) -> Tuple[List[MTMCTrackletCandidate], List[CandidateMotionMetrics]]:
    """Return motion-clean candidates and metrics for every input candidate."""
    clean = []
    metrics_list = []
    for candidate in _progress_iter(candidates, show_progress, "filter motion clean candidates"):
        metrics = compute_candidate_motion_metrics(candidate, config)
        metrics_list.append(metrics)
        if metrics.is_motion_clean:
            clean.append(candidate)
    return clean, metrics_list


def split_candidates_and_metrics(
    candidates: List[MTMCTrackletCandidate],
    config: Dict[str, Any],
    show_progress: bool = True,
) -> Tuple[Dict[str, List[MTMCTrackletCandidate]], List[CandidateMotionMetrics]]:
    """Split candidates and return metrics computed once."""
    buckets = {"clean": [], "suspicious": [], "invalid": [], "unknown": []}
    metrics_list = []
    for candidate in _progress_iter(candidates, show_progress, "motion quality"):
        metrics = compute_candidate_motion_metrics(candidate, config)
        metrics_list.append(metrics)
        if metrics.is_motion_clean:
            buckets["clean"].append(candidate)
        elif metrics.motion_quality_flag == "motion_suspicious":
            buckets["suspicious"].append(candidate)
        elif metrics.motion_quality_flag == "motion_unknown":
            buckets["unknown"].append(candidate)
        else:
            buckets["invalid"].append(candidate)
    return buckets, metrics_list


def _metrics_to_compact_dict(metrics: CandidateMotionMetrics) -> Dict[str, Any]:
    return {
        "motion_quality_flag": metrics.motion_quality_flag,
        "motion_reject_reason": metrics.motion_reject_reason,
        "is_motion_clean": metrics.is_motion_clean,
        "max_step_distance_3d": metrics.max_step_distance_3d,
        "mean_step_distance_3d": metrics.mean_step_distance_3d,
        "median_step_distance_3d": metrics.median_step_distance_3d,
        "p95_step_distance_3d": metrics.p95_step_distance_3d,
        "max_speed_3d": metrics.max_speed_3d,
        "mean_speed_3d": metrics.mean_speed_3d,
        "travel_distance_3d_recomputed": metrics.travel_distance_3d_recomputed,
        "straight_line_distance_3d": metrics.straight_line_distance_3d,
        "path_efficiency_3d": metrics.path_efficiency_3d,
        "travel_distance_per_frame": metrics.travel_distance_per_frame,
        "jump_count": metrics.jump_count,
        "jump_ratio": metrics.jump_ratio,
        "num_valid_3d_points": metrics.num_valid_3d_points,
        "valid_3d_ratio": metrics.valid_3d_ratio,
    }


def _progress_iter(values: List[MTMCTrackletCandidate], show_progress: bool, desc: str) -> Iterable[MTMCTrackletCandidate]:
    if not show_progress:
        return values
    try:
        from tqdm import tqdm
    except ImportError:
        return _print_progress_iter(values, desc)
    return tqdm(values, desc=desc, unit="candidate")


def _print_progress_iter(values: List[MTMCTrackletCandidate], desc: str) -> Iterable[MTMCTrackletCandidate]:
    total = len(values)
    for index, value in enumerate(values):
        if index == 0 or (index + 1) % 1000 == 0 or index + 1 == total:
            print("%s: candidate %d/%d" % (desc, index + 1, total))
        yield value
