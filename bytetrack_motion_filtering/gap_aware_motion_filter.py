"""Gap-aware and class-aware candidate motion classification."""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from deep_oc_sort_3d.mtmc.candidate_motion_quality import (
    CandidateMotionMetrics,
    compute_candidate_motion_metrics,
)
from deep_oc_sort_3d.mtmc.candidate_types import MTMCTrackletCandidate


@dataclass
class GapAwareMotionDecision:
    """Motion decision plus step-level diagnostics for one candidate."""

    metrics: CandidateMotionMetrics
    diagnostics: Dict[str, Any]


def classify_candidate_gap_aware(
    candidate: MTMCTrackletCandidate,
    variant_config: Dict[str, Any],
    priors: Dict[str, Any],
    current_config: Optional[Dict[str, Any]] = None,
) -> GapAwareMotionDecision:
    """Classify one candidate using the requested Step 21E mode."""
    mode = str(variant_config.get("mode", "current"))
    metrics = compute_candidate_motion_metrics(candidate, current_config)
    if mode == "current":
        return GapAwareMotionDecision(metrics=metrics, diagnostics=_diagnostics(candidate, metrics, mode, [], 0, 0))

    min_points = int(variant_config.get("min_valid_3d_points", 3))
    min_ratio = float(variant_config.get("min_valid_3d_ratio", 0.5))
    if metrics.num_valid_3d_points < min_points:
        _set(metrics, "motion_unknown", False, "not_enough_3d_points")
        return GapAwareMotionDecision(metrics, _diagnostics(candidate, metrics, mode, [], 0, 0))
    if metrics.valid_3d_ratio < min_ratio:
        _set(metrics, "motion_invalid", False, "low_valid_3d_ratio")
        return GapAwareMotionDecision(metrics, _diagnostics(candidate, metrics, mode, [], 0, 0))

    prior = _prior_for_candidate(candidate, priors)
    absolute_cap = float(prior.get("absolute_cap", 12.0)) * float(variant_config.get("absolute_cap_multiplier", 1.0))
    bbox_changes = _bbox_change_by_step(candidate)
    violations = []
    extreme_count = 0
    tolerated_count = 0
    for frame_a, frame_b, distance, gap in metrics.step_distances_3d:
        allowed = _allowed_displacement(mode, gap, prior, variant_config, absolute_cap)
        bbox_delta, bbox_ratio = bbox_changes.get((frame_a, frame_b), (None, None))
        is_extreme = float(distance) > absolute_cap
        is_violation = float(distance) > allowed
        is_tolerated = False
        if is_violation and not is_extreme and mode in ("bbox_jump_tolerant", "balanced"):
            is_tolerated = _bbox_jump_supports_tolerance(gap, bbox_delta, bbox_ratio, variant_config)
        if is_extreme:
            extreme_count += 1
        if is_violation:
            violations.append(
                {
                    "frame_a": frame_a,
                    "frame_b": frame_b,
                    "gap": gap,
                    "distance": distance,
                    "allowed": allowed,
                    "absolute_cap": absolute_cap,
                    "bbox_height_delta": bbox_delta,
                    "bbox_height_ratio": bbox_ratio,
                    "tolerated": is_tolerated,
                }
            )
            if is_tolerated:
                tolerated_count += 1

    metrics.jump_count = len(violations)
    metrics.jump_ratio = float(len(violations)) / float(len(metrics.step_distances_3d)) if metrics.step_distances_3d else 0.0
    max_tolerated = int(variant_config.get("max_suspicious_jumps", 1))
    max_jump_ratio = float(variant_config.get("max_accepted_jump_ratio", 0.25))
    untolerated = len(violations) - tolerated_count
    if extreme_count > int(variant_config.get("max_extreme_jumps", 0)):
        _set(metrics, "motion_invalid", False, "absolute_class_cap_exceeded")
    elif untolerated > max_tolerated or metrics.jump_ratio > max_jump_ratio:
        _set(metrics, "motion_invalid", False, "gap_aware_jump_ratio_exceeded")
    elif violations:
        reason = "suspicious_gap_pseudo3d_jump" if tolerated_count else "gap_aware_step_tolerated"
        _set(metrics, "motion_suspicious", True, reason)
    else:
        _set(metrics, "motion_good", True, "ok")
    return GapAwareMotionDecision(
        metrics=metrics,
        diagnostics=_diagnostics(candidate, metrics, mode, violations, extreme_count, tolerated_count),
    )


def _allowed_displacement(
    mode: str,
    gap: int,
    prior: Dict[str, Any],
    config: Dict[str, Any],
    absolute_cap: float,
) -> float:
    if mode == "gap_aware_soft":
        value = float(config.get("base_step_threshold", 6.0)) + float(config.get("gap_factor", 0.8)) * float(gap)
        return min(value, absolute_cap)
    value = float(prior.get("recommended_v_max", prior.get("v_max", 3.0))) * float(gap)
    value += float(prior.get("recommended_margin", prior.get("margin", 1.5)))
    if mode in ("class_aware_strict_cap", "bbox_jump_tolerant", "balanced"):
        return min(value, absolute_cap)
    return value


def _prior_for_candidate(candidate: MTMCTrackletCandidate, priors: Dict[str, Any]) -> Dict[str, Any]:
    classes = priors.get("classes", priors)
    value = classes.get(candidate.class_name, {}) if isinstance(classes, dict) else {}
    return value if isinstance(value, dict) else {}


def _bbox_change_by_step(candidate: MTMCTrackletCandidate) -> Dict[Tuple[int, int], Tuple[Optional[float], Optional[float]]]:
    heights = {}
    for item in candidate.trajectory_2d_sampled:
        if len(item) < 5:
            continue
        heights[int(item[0])] = max(0.0, float(item[4]) - float(item[2]))
    output = {}
    frames = [int(item[0]) for item in candidate.trajectory_3d_sampled if len(item) >= 4]
    for index in range(1, len(frames)):
        frame_a = frames[index - 1]
        frame_b = frames[index]
        height_a = heights.get(frame_a)
        height_b = heights.get(frame_b)
        if height_a is None or height_b is None:
            output[(frame_a, frame_b)] = (None, None)
            continue
        delta = abs(height_b - height_a)
        minimum = max(min(height_a, height_b), 1e-6)
        output[(frame_a, frame_b)] = (delta, max(height_a, height_b) / minimum)
    return output


def _bbox_jump_supports_tolerance(
    gap: int,
    bbox_delta: Optional[float],
    bbox_ratio: Optional[float],
    config: Dict[str, Any],
) -> bool:
    if gap > 1:
        return True
    delta_threshold = float(config.get("bbox_height_delta_threshold", 20.0))
    ratio_threshold = float(config.get("bbox_height_ratio_threshold", 1.35))
    return bool(
        (bbox_delta is not None and bbox_delta >= delta_threshold)
        or (bbox_ratio is not None and bbox_ratio >= ratio_threshold)
    )


def _set(metrics: CandidateMotionMetrics, flag: str, clean: bool, reason: str) -> None:
    metrics.motion_quality_flag = flag
    metrics.is_motion_clean = bool(clean)
    metrics.motion_reject_reason = reason


def _diagnostics(
    candidate: MTMCTrackletCandidate,
    metrics: CandidateMotionMetrics,
    mode: str,
    violations: List[Dict[str, Any]],
    extreme_count: int,
    tolerated_count: int,
) -> Dict[str, Any]:
    gaps = [int(item[3]) for item in metrics.step_distances_3d]
    return {
        "candidate_id": candidate.candidate_id,
        "subset": candidate.subset,
        "scene_name": candidate.scene_name,
        "camera_id": candidate.camera_id,
        "class_id": candidate.class_id,
        "class_name": candidate.class_name,
        "length": candidate.length,
        "duration": candidate.duration,
        "mode": mode,
        "is_motion_clean": metrics.is_motion_clean,
        "motion_quality_flag": metrics.motion_quality_flag,
        "motion_reject_reason": metrics.motion_reject_reason,
        "num_steps": len(metrics.step_distances_3d),
        "max_gap": max(gaps) if gaps else 0,
        "mean_gap": (sum(gaps) / float(len(gaps))) if gaps else None,
        "max_step_distance_3d": metrics.max_step_distance_3d,
        "p95_step_distance_3d": metrics.p95_step_distance_3d,
        "jump_count": metrics.jump_count,
        "jump_ratio": metrics.jump_ratio,
        "extreme_jump_count": extreme_count,
        "tolerated_jump_count": tolerated_count,
        "violation_count": len(violations),
        "violations": violations,
    }

