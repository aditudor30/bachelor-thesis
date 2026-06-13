"""Target recovery at the official scenes and classes with the largest V3 gap."""

from collections import defaultdict
from typing import Any, Dict, List, Sequence, Tuple

from deep_oc_sort_3d.official_023_027.official_track1_io import OfficialTrack1Row
from deep_oc_sort_3d.v3_coverage_extension.recovery_source_loader import RecoveryTrack
from deep_oc_sort_3d.v3_coverage_extension.short_track_recovery import common_reject_reason


def select_scene_class_targeted(
    tracks: Sequence[RecoveryTrack],
    baseline_rows: Sequence[OfficialTrack1Row],
    config: Dict[str, Any],
) -> Tuple[List[RecoveryTrack], Dict[str, Any]]:
    """Select and cap recovery in target scene/class cells."""
    rules = config.get("recovery_rules", {}).get("scene_class_targeted_recovery", {})
    targeting = config.get("targeting", {})
    target_scenes = set(int(value) for value in targeting.get("target_scenes", [24, 26, 23]))
    target_classes = set(int(value) for value in targeting.get("target_official_classes", [0, 1, 3]))
    baseline_scene = defaultdict(int)
    baseline_class = defaultdict(int)
    for row in baseline_rows:
        baseline_scene[int(row.scene_id)] += 1
        baseline_class[int(row.class_id)] += 1
    scene_cap = {key: max(1, int(value * float(rules.get("max_added_rows_ratio_per_scene", 0.75)))) for key, value in baseline_scene.items()}
    class_cap = {key: max(1, int(value * float(rules.get("max_added_rows_ratio_per_class", 0.75)))) for key, value in baseline_class.items()}
    candidates = []
    reasons = {}
    for track in tracks:
        reason = common_reject_reason(track)
        is_target = track.scene_id in target_scenes and track.official_class_id in target_classes
        if reason is None and bool(rules.get("apply_only_to_target_scenes_classes", True)) and not is_target:
            reason = "outside_target_scene_class"
        threshold = float(rules.get("min_mean_confidence_target", 0.28) if is_target else rules.get("min_mean_confidence_non_target", 0.45))
        if reason is None and track.mean_confidence < threshold:
            reason = "confidence_too_low"
        if reason is None and track.length < int(rules.get("min_length", 3)):
            reason = "too_short"
        if reason is None and track.p95_step_distance is not None and track.p95_step_distance > float(rules.get("max_step_p95_m", 18.0)):
            reason = "motion_p95_too_large"
        if reason is None:
            candidates.append(track)
        else:
            reasons[str(reason)] = reasons.get(str(reason), 0) + 1
    candidates = sorted(candidates, key=lambda item: (item.mean_confidence, item.length), reverse=True)
    selected = []
    used_scene = defaultdict(int)
    used_class = defaultdict(int)
    for track in candidates:
        if used_scene[track.scene_id] + track.length > scene_cap.get(track.scene_id, track.length):
            reasons["scene_cap_reached"] = reasons.get("scene_cap_reached", 0) + 1
            continue
        if used_class[track.official_class_id] + track.length > class_cap.get(track.official_class_id, track.length):
            reasons["class_cap_reached"] = reasons.get("class_cap_reached", 0) + 1
            continue
        selected.append(track)
        used_scene[track.scene_id] += track.length
        used_class[track.official_class_id] += track.length
    return selected, {
        "selected_tracks": len(selected), "selected_rows": sum(item.length for item in selected),
        "selected_rows_by_scene": dict(sorted(used_scene.items())), "selected_rows_by_class": dict(sorted(used_class.items())),
        "reject_reasons": reasons,
    }

