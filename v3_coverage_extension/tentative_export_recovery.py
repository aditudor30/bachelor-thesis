"""Recovery of tentative or low-confidence detection-associated ByteTrack states."""

from typing import Any, Dict, List, Sequence, Tuple

from deep_oc_sort_3d.v3_coverage_extension.recovery_source_loader import RecoveryTrack
from deep_oc_sort_3d.v3_coverage_extension.short_track_recovery import common_reject_reason


def select_associated_tentative(tracks: Sequence[RecoveryTrack], config: Dict[str, Any]) -> Tuple[List[RecoveryTrack], Dict[str, Any]]:
    """Select real-detection tracks omitted by confirmed-track export policy."""
    rules = config.get("recovery_rules", {}).get("associated_tentative_export", {})
    allowed_states = set(str(value).lower() for value in rules.get("allowed_states", ["tentative", "unconfirmed", "lost"]))
    selected = []
    reasons = {}
    for track in tracks:
        reason = common_reject_reason(track)
        if reason is None and track.mean_confidence < float(rules.get("min_mean_confidence", 0.30)):
            reason = "confidence_too_low"
        if reason is None and track.length > int(rules.get("max_length", 15)):
            reason = "track_too_long_for_tentative_recovery"
        state_match = bool(track.states.intersection(allowed_states))
        low_confirmed = bool(rules.get("allow_low_confidence_confirmed", True)) and "confirmed" in track.states and track.mean_confidence <= float(rules.get("max_confirmed_mean_confidence", 0.45))
        if reason is None and not state_match and not low_confirmed:
            reason = "not_tentative_or_low_confidence_confirmed"
        if reason is None:
            selected.append(track)
        else:
            reasons[str(reason)] = reasons.get(str(reason), 0) + 1
    return selected, {"selected_tracks": len(selected), "selected_rows": sum(item.length for item in selected), "reject_reasons": reasons}

