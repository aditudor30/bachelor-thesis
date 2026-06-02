"""Quality scoring and filtering for local tracklets."""

from typing import Optional, Tuple

from deep_oc_sort_3d.tracklets.tracklet_types import LocalTracklet


def compute_tracklet_quality_score(
    tracklet: LocalTracklet,
    min_length: int = 3,
    min_mean_confidence: float = 0.01,
    prefer_3d: bool = True,
) -> float:
    """Compute a compact 0..1 quality score for MTMC-prep diagnostics."""
    length_score = min(float(tracklet.length) / max(float(min_length) * 5.0, 1.0), 1.0)
    confidence_score = min(float(tracklet.mean_confidence) / max(float(min_mean_confidence) * 5.0, 1e-6), 1.0)
    bbox_score = 1.0 if tracklet.trajectory_2d else 0.0
    center_score = 1.0 if tracklet.trajectory_3d else 0.0
    span = max(int(tracklet.end_frame) - int(tracklet.start_frame) + 1, 1)
    continuity_score = min(float(tracklet.length) / float(span), 1.0)
    score = 0.30 * length_score + 0.25 * confidence_score + 0.20 * bbox_score + 0.15 * continuity_score
    if prefer_3d:
        score += 0.10 * center_score
    else:
        score += 0.10
    return max(0.0, min(float(score), 1.0))


def classify_tracklet_quality(
    tracklet: LocalTracklet,
    min_length: int = 3,
    min_mean_confidence: float = 0.01,
    min_gt_purity: Optional[float] = None,
) -> Tuple[str, bool, str]:
    """Classify tracklet quality and decide whether it is MTMC-ready."""
    notes = []
    is_valid = True
    flag = "good"

    if int(tracklet.class_id) < 0 or not tracklet.trajectory_2d:
        flag = "invalid"
        is_valid = False
        notes.append("missing class id or 2d trajectory")
    elif int(tracklet.length) < int(min_length):
        flag = "short"
        is_valid = False
        notes.append("length below min_length")
    elif float(tracklet.mean_confidence) < float(min_mean_confidence):
        flag = "low_confidence"
        is_valid = False
        notes.append("mean confidence below threshold")
    else:
        span = max(int(tracklet.end_frame) - int(tracklet.start_frame) + 1, 1)
        observed_ratio = float(tracklet.length) / float(span)
        if min_gt_purity is not None and tracklet.gt_purity is not None and tracklet.gt_purity < float(min_gt_purity):
            flag = "mixed_gt"
            notes.append("gt purity below diagnostic threshold")
        elif not tracklet.trajectory_3d:
            flag = "no_3d"
            notes.append("missing 3d trajectory")
        elif observed_ratio < 0.5:
            flag = "fragmented"
            notes.append("large temporal gaps in tracklet")

    if tracklet.notes:
        notes.append(tracklet.notes)
    return flag, is_valid, "; ".join(notes)
