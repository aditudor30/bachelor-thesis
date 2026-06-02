"""Filtering and scoring rules for MTMC tracklet candidates."""

from typing import List, Optional, Tuple

from deep_oc_sort_3d.tracklets.tracklet_types import LocalTracklet


DEFAULT_ALLOWED_QUALITY_FLAGS = ["good", "fragmented"]


def should_keep_tracklet(
    tracklet: LocalTracklet,
    min_length: int = 3,
    min_mean_confidence: float = 0.01,
    allowed_quality_flags: Optional[List[str]] = None,
    require_valid_for_mtmc: bool = True,
    require_3d: bool = False,
    class_allowlist: Optional[List[str]] = None,
    class_blocklist: Optional[List[str]] = None,
) -> Tuple[bool, str]:
    """Return whether a tracklet should become an MTMC candidate."""
    allowed = DEFAULT_ALLOWED_QUALITY_FLAGS if allowed_quality_flags is None else list(allowed_quality_flags)
    if class_blocklist is not None and tracklet.class_name in set(class_blocklist):
        return False, "class_blocked"
    if class_allowlist is not None and tracklet.class_name not in set(class_allowlist):
        return False, "class_not_allowed"
    if int(tracklet.length) < int(min_length):
        return False, "too_short"
    if float(tracklet.mean_confidence) < float(min_mean_confidence):
        return False, "low_confidence"
    if str(tracklet.quality_flag) not in set(allowed):
        return False, "quality_flag_not_allowed"
    if require_valid_for_mtmc and not bool(tracklet.is_valid_for_mtmc):
        return False, "invalid_for_mtmc"
    if require_3d and not bool(tracklet.trajectory_3d):
        return False, "missing_3d"
    return True, "ok"


def compute_candidate_quality_score(tracklet: LocalTracklet) -> float:
    """Compute a candidate quality score without using GT diagnostics."""
    length_score = min(float(tracklet.length) / 30.0, 1.0)
    confidence_score = max(0.0, min(float(tracklet.mean_confidence), 1.0))
    flag_score = _flag_score(str(tracklet.quality_flag))
    has_3d_score = 1.0 if tracklet.trajectory_3d else 0.0
    smoothness_score = _trajectory_smoothness_score(tracklet)
    score = (
        0.30 * length_score
        + 0.25 * confidence_score
        + 0.20 * flag_score
        + 0.15 * has_3d_score
        + 0.10 * smoothness_score
    )
    return max(0.0, min(float(score), 1.0))


def _flag_score(flag: str) -> float:
    if flag == "good":
        return 1.0
    if flag == "fragmented":
        return 0.7
    if flag == "no_3d":
        return 0.55
    if flag == "short":
        return 0.25
    if flag == "low_confidence":
        return 0.2
    return 0.0


def _trajectory_smoothness_score(tracklet: LocalTracklet) -> float:
    trajectory = tracklet.trajectory_3d
    if len(trajectory) < 3:
        return 0.5 if trajectory else 0.0
    distances = []
    for index in range(1, len(trajectory)):
        prev = trajectory[index - 1]
        cur = trajectory[index]
        dx = float(cur[1]) - float(prev[1])
        dy = float(cur[2]) - float(prev[2])
        dz = float(cur[3]) - float(prev[3])
        distances.append((dx * dx + dy * dy + dz * dz) ** 0.5)
    if not distances:
        return 0.5
    mean_dist = sum(distances) / float(len(distances))
    if mean_dist <= 1e-6:
        return 1.0
    variance = sum((item - mean_dist) ** 2 for item in distances) / float(len(distances))
    normalized = variance / max(mean_dist * mean_dist, 1e-6)
    return max(0.0, min(1.0, 1.0 / (1.0 + normalized)))
