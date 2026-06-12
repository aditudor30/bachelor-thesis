"""Select a local tracker candidate using fragmentation and purity safeguards."""

from typing import Any, Dict, List, Optional


def select_local_tracker(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Select a candidate only when track continuity improves safely."""
    baseline = _find(rows, "current_local_tracker")
    successful = [row for row in rows if row.get("status") == "ok"]
    if baseline is None or not successful:
        return {"selected_tracker": None, "verdict": "benchmark_invalid_fix_required"}
    candidates = []
    for row in successful:
        if row.get("tracker_name") == "current_local_tracker":
            continue
        reasons = candidate_rejection_reasons(row, baseline)
        candidate = dict(row)
        candidate["selection_reasons"] = reasons
        if not reasons:
            candidates.append(candidate)
    if not candidates:
        return {"selected_tracker": "current_local_tracker", "verdict": "current_tracker_still_best"}
    candidates.sort(key=lambda row: _selection_key(row, baseline))
    selected = candidates[0]
    name = str(selected.get("tracker_name"))
    if name == "bytetrack_style_yolo11m":
        verdict = "bytetrack_style_candidate_for_full_rerun"
    elif name == "botsort_osnet_finetuned_yolo11m":
        verdict = "botsort_osnet_candidate_for_full_rerun"
    elif "sbs" in name:
        verdict = "botsort_sbs_candidate_for_full_rerun"
    else:
        verdict = "benchmark_inconclusive"
    return {"selected_tracker": name, "verdict": verdict, "selected_metrics": selected}


def candidate_rejection_reasons(row: Dict[str, Any], baseline: Dict[str, Any]) -> List[str]:
    """Reject shorter, more fragmented, impure or non-Person-degrading variants."""
    reasons = []
    if _number(row.get("median_track_length")) <= _number(baseline.get("median_track_length")):
        reasons.append("median_track_length_not_improved")
    if _number(row.get("short_track_ratio_le3")) >= _number(baseline.get("short_track_ratio_le3")):
        reasons.append("short_track_ratio_not_improved")
    purity = _optional_number(row.get("local_purity_mean"))
    baseline_purity = _optional_number(baseline.get("local_purity_mean"))
    if purity is not None and baseline_purity is not None and baseline_purity - purity > 0.02:
        reasons.append("purity_drop")
    false_merge = _optional_number(row.get("false_merge_suspicion_rate"))
    baseline_false = _optional_number(baseline.get("false_merge_suspicion_rate"))
    if false_merge is not None and baseline_false is not None and false_merge - baseline_false > 0.02:
        reasons.append("false_merge_increase")
    non_person = _optional_number(row.get("nonperson_short_track_ratio_le3"))
    baseline_non = _optional_number(baseline.get("nonperson_short_track_ratio_le3"))
    if non_person is not None and baseline_non is not None and non_person - baseline_non > 0.05:
        reasons.append("non_person_degradation")
    return reasons


def _selection_key(row: Dict[str, Any], baseline: Dict[str, Any]) -> Any:
    return (
        _number(row.get("short_track_ratio_le3")),
        -_number(row.get("median_track_length")),
        -_number(row.get("mean_track_length")),
        _number(row.get("runtime_seconds")),
    )


def _find(rows: List[Dict[str, Any]], name: str) -> Optional[Dict[str, Any]]:
    for row in rows:
        if row.get("tracker_name") == name:
            return row
    return None


def _number(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("inf")


def _optional_number(value: Any) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
