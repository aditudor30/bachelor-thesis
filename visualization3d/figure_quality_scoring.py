"""Heuristic scoring for MVP figure candidates."""

from typing import Any, Dict, List, Optional, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from deep_oc_sort_3d.visualization3d.figure_candidate_selection import FigureCandidate


def score_tracking_2d_candidate(candidate: "FigureCandidate") -> float:
    """Score frame-level 2D tracking candidates for clean demo figures."""
    if candidate.num_records <= 0:
        return 0.0
    assigned_ratio = float(candidate.num_assigned) / float(candidate.num_records)
    density_score = _moderate_count_score(candidate.num_records, target=12, spread=18)
    class_score = min(1.0, float(candidate.num_classes) / 3.0)
    clutter_penalty = max(0.0, float(candidate.num_records - 30) / 30.0)
    return float(5.0 * density_score + 3.0 * assigned_ratio + 2.0 * class_score - 2.0 * clutter_penalty)


def score_cuboid_3d_candidate(candidate: "FigureCandidate") -> float:
    """Score cuboid diagnostic candidates."""
    if candidate.num_records <= 0:
        return 0.0
    projection_ratio = candidate.projection_success_rate
    if projection_ratio is None:
        projection_ratio = min(1.0, float(candidate.num_projectable_3d) / float(max(candidate.num_records, 1)))
    density_score = _moderate_count_score(candidate.num_records, target=8, spread=16)
    cuboid_score = min(1.0, float(candidate.num_projectable_3d) / 6.0)
    return float(4.0 * density_score + 4.0 * cuboid_score + 4.0 * float(projection_ratio))


def score_bev_candidate(
    records: List[Dict[str, Any]],
    min_track_length: int = 5,
    max_tracks: int = 100,
) -> float:
    """Score a generic export scene for coordinate-space BEV visualization."""
    groups = {}
    classes = set()
    for record in records:
        track_id = _optional_int(record.get("global_track_id"))
        if track_id is None:
            continue
        groups.setdefault(track_id, 0)
        groups[track_id] += 1
        class_name = str(record.get("class_name", ""))
        if class_name:
            classes.add(class_name)
    lengths = [length for length in groups.values() if length >= int(min_track_length)]
    if not lengths:
        return 0.0
    mean_length = float(np.mean(np.asarray(lengths, dtype=float)))
    track_count_score = _moderate_count_score(min(len(lengths), int(max_tracks)), target=40, spread=80)
    length_score = min(1.0, mean_length / 50.0)
    class_score = min(1.0, float(len(classes)) / 4.0)
    return float(4.0 * track_count_score + 3.0 * length_score + 2.0 * class_score)


def explain_candidate_score(candidate: "FigureCandidate") -> str:
    """Return a short human-readable explanation for a candidate score."""
    if candidate.figure_type == "cuboid_3d":
        return (
            "score favors moderate crowding, projected cuboids, and high projection success; "
            "records=%d projectable_3d=%d projection_success_rate=%s"
            % (candidate.num_records, candidate.num_projectable_3d, str(candidate.projection_success_rate))
        )
    return (
        "score favors moderate crowding, assigned global IDs, and class diversity; "
        "records=%d assigned=%d classes=%d"
        % (candidate.num_records, candidate.num_assigned, candidate.num_classes)
    )


def _moderate_count_score(count: int, target: int, spread: int) -> float:
    distance = abs(float(count) - float(target))
    return max(0.0, 1.0 - distance / float(max(spread, 1)))


def _optional_int(value: Any) -> Optional[int]:
    if value in (None, ""):
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None

