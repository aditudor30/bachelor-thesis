"""Preserve clean V3 tracks as independent single-camera identities."""

from typing import Any, Dict, List, Sequence, Tuple

from deep_oc_sort_3d.v3_coverage_extension.recovery_source_loader import RecoveryTrack
from deep_oc_sort_3d.v3_coverage_extension.short_track_recovery import common_reject_reason


def select_single_camera_clean(tracks: Sequence[RecoveryTrack], config: Dict[str, Any]) -> Tuple[List[RecoveryTrack], Dict[str, Any]]:
    """Select clean unrepresented tracks without forcing global merges."""
    rules = config.get("recovery_rules", {}).get("single_camera_keep_clean", {})
    selected = []
    reasons = {}
    for track in tracks:
        reason = common_reject_reason(track)
        if reason is None and track.length < int(rules.get("min_length", 5)):
            reason = "too_short"
        threshold = float(rules.get("min_mean_confidence_rare", 0.45) if track.official_class_id in (2, 4, 5, 6) else rules.get("min_mean_confidence", 0.35))
        if reason is None and track.mean_confidence < threshold:
            reason = "confidence_too_low"
        if reason is None and track.p95_step_distance is not None and track.p95_step_distance > float(rules.get("max_step_p95_m", 12.0)):
            reason = "motion_p95_too_large"
        quality_values = set([track.tracklet_quality_flag.lower(), track.candidate_quality_flag.lower()])
        if reason is None and bool(rules.get("require_smoothness_good_or_unknown", True)) and not quality_values.intersection(set(["good", "ok", "unknown", "clean", "valid"])):
            reason = "quality_not_good_or_unknown"
        if reason is None and bool(rules.get("require_no_extreme_jumps", True)) and track.jump_ratio is not None and track.jump_ratio > 0.05:
            reason = "extreme_jump_ratio"
        if reason is None:
            selected.append(track)
        else:
            reasons[str(reason)] = reasons.get(str(reason), 0) + 1
    return selected, {"selected_tracks": len(selected), "selected_rows": sum(item.length for item in selected), "reject_reasons": reasons}

