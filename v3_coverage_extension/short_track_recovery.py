"""Safe recovery of short, detection-backed V3 tracks."""

from typing import Any, Dict, List, Sequence, Tuple

from deep_oc_sort_3d.v3_coverage_extension.recovery_source_loader import RecoveryTrack


def select_short_track_safe(tracks: Sequence[RecoveryTrack], config: Dict[str, Any]) -> Tuple[List[RecoveryTrack], Dict[str, Any]]:
    """Select short tracks with complete geometry and plausible motion."""
    rules = config.get("recovery_rules", {}).get("short_track_safe", {})
    selected = []
    reasons = {}
    for track in tracks:
        reason = _short_reject_reason(track, rules)
        if reason is None:
            selected.append(track)
        else:
            reasons[reason] = reasons.get(reason, 0) + 1
    return selected, {"selected_tracks": len(selected), "selected_rows": sum(item.length for item in selected), "reject_reasons": reasons}


def common_reject_reason(track: RecoveryTrack, require_all_geometry: bool = True) -> Any:
    """Apply source-integrity checks shared by all recovery mechanisms."""
    if track.baseline_covered:
        return "already_covered_by_v3"
    if track.length <= 0:
        return "empty_track"
    if require_all_geometry and track.geometry_valid_count != track.length:
        return "incomplete_geometry"
    if not require_all_geometry and track.geometry_valid_count <= 0:
        return "no_valid_geometry"
    if any(int(record.detection_id) < 0 for record in track.records):
        return "not_detection_associated"
    return None


def confidence_threshold(track: RecoveryTrack, rules: Dict[str, Any]) -> float:
    """Return class-aware confidence threshold using internal class IDs."""
    if track.internal_class_id == 0:
        return float(rules.get("min_mean_confidence_person", rules.get("min_mean_confidence_default", 0.35)))
    if track.internal_class_id == 1:
        return float(rules.get("min_mean_confidence_forklift", rules.get("min_mean_confidence_default", 0.35)))
    if track.official_class_id in (2, 4, 5, 6):
        return float(rules.get("min_mean_confidence_rare", rules.get("min_mean_confidence_default", 0.35)))
    return float(rules.get("min_mean_confidence_default", 0.35))


def _short_reject_reason(track: RecoveryTrack, rules: Dict[str, Any]) -> Any:
    common = common_reject_reason(track)
    if common is not None:
        return common
    minimum = int(rules.get("min_length_default", 3))
    if track.internal_class_id == 0:
        minimum = int(rules.get("min_length_person", minimum))
    elif track.internal_class_id == 1:
        minimum = int(rules.get("min_length_forklift", minimum))
    if track.length < minimum:
        return "too_short"
    if track.length > int(rules.get("max_length", 8)):
        return "not_short_track"
    if track.mean_confidence < confidence_threshold(track, rules):
        return "confidence_too_low"
    if track.p95_step_distance is not None and track.p95_step_distance > float(rules.get("max_step_p95_m", 15.0)):
        return "motion_p95_too_large"
    return None

