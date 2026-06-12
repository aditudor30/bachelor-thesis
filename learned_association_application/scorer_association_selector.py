"""Select a safe learned association sweep variant."""

from typing import Any, Dict, List, Optional

from deep_oc_sort_3d.learned_association_application.scorer_association_io import safe_float


def select_variant(
    rows: List[Dict[str, Any]],
    baseline: Dict[str, Any],
    selection: Dict[str, Any],
) -> Dict[str, Any]:
    """Select by validity, class preservation, false merges, purity and fragmentation."""
    valid = []
    for row in rows:
        reasons = selection_reasons(row, baseline, selection)
        candidate = dict(row)
        candidate["selection_reasons"] = reasons
        if not reasons:
            valid.append(candidate)
    if not valid:
        verdict = "mlp_association_invalid_fix_required" if not rows else _failure_verdict(rows, baseline, selection)
        return {"selected_variant": None, "verdict": verdict, "candidates": rows}
    valid.sort(key=lambda row: _selection_key(row, baseline))
    selected = valid[0]
    fragmentation_gain = _delta(baseline.get("person_fragmentation"), selected.get("person_fragmentation"))
    clear_gain = float(selection.get("min_person_fragmentation_reduction_for_clear_gain", 60))
    verdict = "mlp_association_improves_over_combined_safe_080" if fragmentation_gain is not None and fragmentation_gain >= clear_gain else "mlp_association_valid_small_gain"
    if fragmentation_gain is None or fragmentation_gain <= 0:
        verdict = "mlp_association_valid_but_no_clear_gain"
    return {
        "selected_variant": selected.get("run_name"),
        "verdict": verdict,
        "selected_metrics": selected,
        "person_fragmentation_reduction": fragmentation_gain,
        "candidates_passing_selection": len(valid),
    }


def selection_reasons(
    row: Dict[str, Any], baseline: Dict[str, Any], selection: Dict[str, Any]
) -> List[str]:
    """Return hard selection failures for one variant."""
    reasons = []
    if bool(selection.get("require_track1_valid", True)) and not bool(row.get("track1_valid")):
        reasons.append("track1_invalid")
    non_person_delta = _delta(row.get("non_person_rows"), baseline.get("non_person_rows"))
    if bool(selection.get("require_non_person_unchanged", True)) and non_person_delta not in (None, 0.0):
        reasons.append("non_person_changed")
    false_delta = _delta(row.get("person_false_merge_rate"), baseline.get("person_false_merge_rate"))
    if false_delta is not None and false_delta > float(selection.get("max_allowed_false_merge_rate_delta", 0.01)):
        reasons.append("false_merge_increase")
    purity_delta = _delta(baseline.get("person_purity_mean"), row.get("person_purity_mean"))
    if purity_delta is not None and purity_delta > float(selection.get("max_allowed_purity_drop", 0.01)):
        reasons.append("purity_drop")
    return reasons


def _selection_key(row: Dict[str, Any], baseline: Dict[str, Any]) -> Any:
    false_delta = _delta(row.get("person_false_merge_rate"), baseline.get("person_false_merge_rate")) or 0.0
    purity_drop = _delta(baseline.get("person_purity_mean"), row.get("person_purity_mean")) or 0.0
    fragmentation = safe_float(row.get("person_fragmentation"), float("inf"))
    rows = safe_float(row.get("track1_rows"), float("inf"))
    return (max(0.0, false_delta), max(0.0, purity_drop), fragmentation, rows)


def _failure_verdict(rows: List[Dict[str, Any]], baseline: Dict[str, Any], selection: Dict[str, Any]) -> str:
    false_failures = 0
    for row in rows:
        reasons = selection_reasons(row, baseline, selection)
        if "false_merge_increase" in reasons:
            false_failures += 1
    if false_failures == len(rows):
        return "mlp_association_increases_false_merges"
    return "mlp_association_valid_but_no_clear_gain"


def _delta(left: Any, right: Any) -> Optional[float]:
    left_value = safe_float(left, None)
    right_value = safe_float(right, None)
    return None if left_value is None or right_value is None else left_value - right_value
