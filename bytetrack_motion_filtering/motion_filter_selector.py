"""Select the safest useful Step 21E motion-filter variant."""

from typing import Any, Dict, List, Optional, Tuple

from deep_oc_sort_3d.bytetrack_motion_filtering.motion_filter_config import output_root
from deep_oc_sort_3d.bytetrack_motion_filtering.motion_filter_io import write_json


def select_motion_filter_variant(config: Dict[str, Any], comparison: Dict[str, Any]) -> Dict[str, Any]:
    """Apply hard safety gates, then rank coverage and continuity gains."""
    variants = comparison.get("variants", {})
    baselines = comparison.get("baselines", {})
    bt21c = baselines.get("bytetrack_21c_best", {})
    current = variants.get("current_motion_filter", {})
    if current.get("status") != "ok":
        current = bt21c
    v2 = baselines.get("baseline_v2_current", {})
    selection = config.get("selection", {})
    evaluated = []
    for name, value in variants.items():
        if name == "current_motion_filter":
            continue
        failures = _hard_failures(value, current, bt21c, selection)
        score = _score(value, current, bt21c, v2)
        evaluated.append({"variant_name": name, "hard_criteria_met": not failures, "hard_failures": failures, "selection_score": score})
    valid = [row for row in evaluated if row["hard_criteria_met"]]
    valid.sort(key=lambda row: float(row.get("selection_score", -1e9)), reverse=True)
    selected_name = valid[0]["variant_name"] if valid else None
    selected = variants.get(selected_name, {}) if selected_name else {}
    verdict, reasons = _verdict(selected_name, selected, current, bt21c, selection, evaluated)
    result = {
        "selected_variant": selected_name,
        "verdict": verdict,
        "reasons": reasons,
        "evaluated_variants": evaluated,
        "selected_metrics": selected,
        "recommended_step_21f": _step_21f(verdict),
    }
    root = output_root(config) / "comparison"
    write_json(root / "selected_motion_filter_variant.json", result)
    write_json(root / "verdict.json", {"label": verdict, "reasons": reasons, "recommended_step_21f": result["recommended_step_21f"]})
    return result


def _hard_failures(
    value: Dict[str, Any],
    current: Dict[str, Any],
    bt21c: Dict[str, Any],
    config: Dict[str, Any],
) -> List[str]:
    failures = []
    track1 = value.get("track1", {})
    if bool(config.get("require_track1_errors_zero", True)) and track1.get("validation_errors") not in (0, "0"):
        failures.append("track1_validation_errors")
    if not _greater(value.get("motion", {}).get("motion_clean_retention"), current.get("motion", {}).get("motion_clean_retention")):
        failures.append("motion_clean_retention_not_improved")
    if not _greater(track1.get("rows"), bt21c.get("track1", {}).get("rows")):
        failures.append("track1_rows_not_above_bytetrack_21c")
    purity_drop = _delta(bt21c.get("global", {}).get("global_purity_mean"), value.get("global", {}).get("global_purity_mean"))
    if purity_drop is not None and purity_drop < -float(config.get("max_allowed_purity_drop", 0.01)):
        failures.append("purity_drop_too_large")
    false_merge_delta = _delta(bt21c.get("global", {}).get("false_merge_rate"), value.get("global", {}).get("false_merge_rate"))
    if false_merge_delta is not None and false_merge_delta > float(config.get("max_allowed_false_merge_rate_delta", 0.01)):
        failures.append("false_merge_rate_increase_too_large")
    return failures


def _score(value: Dict[str, Any], current: Dict[str, Any], bt21c: Dict[str, Any], v2: Dict[str, Any]) -> float:
    motion_gain = _safe(value.get("motion", {}).get("motion_clean_retention")) - _safe(current.get("motion", {}).get("motion_clean_retention"))
    track1_gain = _retention(value.get("track1", {}).get("rows"), bt21c.get("track1", {}).get("rows")) - 1.0
    multi_gain = _retention(value.get("global", {}).get("multi_camera_tracks"), bt21c.get("global", {}).get("multi_camera_tracks")) - 1.0
    purity_delta = _safe(value.get("global", {}).get("global_purity_mean")) - _safe(bt21c.get("global", {}).get("global_purity_mean"))
    false_merge_delta = _safe(value.get("global", {}).get("false_merge_rate")) - _safe(bt21c.get("global", {}).get("false_merge_rate"))
    fragmentation_bonus = _retention(v2.get("global", {}).get("fragmentation_approx"), value.get("global", {}).get("fragmentation_approx")) - 1.0
    return 4.0 * motion_gain + 3.0 * track1_gain + multi_gain + purity_delta - 2.0 * false_merge_delta + 0.25 * fragmentation_bonus


def _verdict(
    selected_name: Optional[str],
    selected: Dict[str, Any],
    current: Dict[str, Any],
    bt21c: Dict[str, Any],
    config: Dict[str, Any],
    evaluated: List[Dict[str, Any]],
) -> Tuple[str, List[str]]:
    if any(value.get("status") in ("error", "invalid") for value in [selected] if value):
        return "gap_aware_motion_filter_invalid_fix_required", ["selected_variant_invalid"]
    if selected_name is None:
        if any("false_merge_rate_increase_too_large" in row.get("hard_failures", []) for row in evaluated):
            return "gap_aware_motion_filter_valid_but_false_merges_too_high", ["coverage_candidates_failed_false_merge_gate"]
        return "gap_aware_motion_filter_no_clear_gain", ["no_variant_met_all_hard_criteria"]
    motion_gain = _safe(selected.get("motion", {}).get("motion_clean_retention")) - _safe(current.get("motion", {}).get("motion_clean_retention"))
    track1_gain = _retention(selected.get("track1", {}).get("rows"), bt21c.get("track1", {}).get("rows")) - 1.0
    min_motion = float(config.get("min_motion_clean_retention_gain", 0.10))
    min_track1 = float(config.get("min_track1_rows_retention_gain", 0.10))
    if motion_gain >= min_motion and track1_gain >= min_track1:
        return "gap_aware_motion_filter_ready_for_v3_candidate", ["motion_and_track1_coverage_improved_with_safety_gates"]
    if motion_gain > 0.0 and track1_gain > 0.0:
        return "gap_aware_motion_filter_valid_improves_coverage_needs_global_tuning", ["coverage_improved_but_below_target_gain"]
    return "gap_aware_motion_filter_valid_but_small_gain", ["safe_gain_is_small"]


def _step_21f(verdict: str) -> str:
    if verdict == "gap_aware_motion_filter_ready_for_v3_candidate":
        return "Step 21F: freeze selected filter and run V3 submission-candidate validation"
    if verdict == "gap_aware_motion_filter_valid_improves_coverage_needs_global_tuning":
        return "Step 21F: tune global association for the selected motion-clean set"
    if verdict == "gap_aware_motion_filter_valid_but_false_merges_too_high":
        return "Step 21F: tighten class caps and global merge safety before full rerun"
    return "Step 21F is not justified yet; refine motion thresholds from diagnostics"


def _greater(a: Any, b: Any) -> bool:
    try:
        return float(a) > float(b)
    except (TypeError, ValueError):
        return False


def _delta(a: Any, b: Any) -> Optional[float]:
    try:
        return float(b) - float(a)
    except (TypeError, ValueError):
        return None


def _retention(a: Any, b: Any) -> float:
    try:
        denominator = float(b)
        return 0.0 if denominator <= 0.0 else float(a) / denominator
    except (TypeError, ValueError):
        return 0.0


def _safe(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0
