"""Selection logic for fine-tuned Person ReID association variants."""

from typing import Any, Dict, List, Optional

from deep_oc_sort_3d.reid_finetuned_association.finetuned_association_io import safe_float


def select_finetuned_reid_variant(comparison: Dict[str, Any], criteria: Dict[str, Any]) -> Dict[str, Any]:
    """Select the safest useful fine-tuned ReID variant, if any."""
    baseline = comparison.get("baseline", {})
    runs = comparison.get("runs", [])
    evaluated = []
    for run in runs:
        row = dict(run)
        row["track1_valid"] = is_track1_valid(row)
        row["non_person_unchanged"] = non_person_unchanged(row)
        row["selection_acceptable"] = acceptable_variant(row, criteria)
        row["selection_score"] = selection_score(row)
        evaluated.append(row)
    invalid = [row for row in evaluated if not row.get("track1_valid")]
    if invalid and len(invalid) == len(evaluated):
        verdict = "finetuned_reid_invalid_fix_required"
        best = None
    elif not any(has_reid_activity(row) for row in evaluated):
        verdict = "finetuned_reid_valid_but_no_clear_gain"
        best = None
    else:
        acceptable = [row for row in evaluated if row.get("selection_acceptable")]
        if acceptable:
            best = sorted(acceptable, key=lambda item: float(item.get("selection_score", -1e9)), reverse=True)[0]
            frag_delta = safe_float(best.get("vs_v2_person_fragmentation_approx_delta"), 0.0) or 0.0
            min_clear = float(criteria.get("min_person_fragmentation_reduction_for_clear_gain", 50))
            if frag_delta <= -min_clear:
                verdict = "finetuned_reid_association_improves_v2"
            else:
                verdict = "finetuned_reid_valid_small_gain"
        else:
            risky = any(is_track1_valid(row) and false_merge_delta(row) > float(criteria.get("max_allowed_false_merge_rate_delta", 0.01)) for row in evaluated)
            best = None
            verdict = "finetuned_reid_increases_false_merges" if risky else "finetuned_reid_valid_but_no_clear_gain"
    selected = {
        "verdict": verdict,
        "best_run": None if best is None else best.get("run_name"),
        "baseline_run": baseline.get("run_name", "v2_current"),
        "criteria": criteria,
        "runs": evaluated,
    }
    return selected


def acceptable_variant(row: Dict[str, Any], criteria: Dict[str, Any]) -> bool:
    """Return whether a variant passes safety gates."""
    if bool(criteria.get("require_reid_activity", True)) and not has_reid_activity(row):
        return False
    if bool(criteria.get("require_track1_valid", True)) and not is_track1_valid(row):
        return False
    if bool(criteria.get("require_non_person_unchanged", True)) and not non_person_unchanged(row):
        return False
    if purity_drop(row) > float(criteria.get("max_allowed_purity_drop", 0.01)):
        return False
    if false_merge_delta(row) > float(criteria.get("max_allowed_false_merge_rate_delta", 0.01)):
        return False
    frag_delta = safe_float(row.get("vs_v2_person_fragmentation_approx_delta"), None)
    if frag_delta is None or frag_delta >= 0.0:
        return False
    return True


def selection_score(row: Dict[str, Any]) -> float:
    """Score acceptable runs; higher is better."""
    frag_gain = -(safe_float(row.get("vs_v2_person_fragmentation_approx_delta"), 0.0) or 0.0)
    purity_penalty = max(0.0, purity_drop(row)) * 10000.0
    false_penalty = max(0.0, false_merge_delta(row)) * 10000.0
    row_penalty = abs(safe_float(row.get("vs_v2_track1_rows_delta"), 0.0) or 0.0) * 0.001
    return frag_gain - purity_penalty - false_penalty - row_penalty


def is_track1_valid(row: Dict[str, Any]) -> bool:
    """Check Track1 validity."""
    errors = row.get("track1_validation_errors", row.get("track1_errors"))
    status = str(row.get("track1_validation_status", "")).lower()
    return errors in (0, 0.0, "0") and status in ("", "ok", "none")


def non_person_unchanged(row: Dict[str, Any]) -> bool:
    """Check non-Person row delta."""
    delta = safe_float(row.get("vs_v2_non_person_rows_delta"), None)
    return delta is not None and abs(float(delta)) <= 0.0


def purity_drop(row: Dict[str, Any]) -> float:
    """Return positive purity drop."""
    delta = safe_float(row.get("vs_v2_global_purity_mean_delta"), 0.0) or 0.0
    return max(0.0, -float(delta))


def false_merge_delta(row: Dict[str, Any]) -> float:
    """Return false merge rate delta."""
    return safe_float(row.get("vs_v2_false_merge_rate_delta"), 0.0) or 0.0


def has_reid_activity(row: Dict[str, Any]) -> bool:
    """Return True only when fine-tuned ReID actually affected or could affect association."""
    with_reid = safe_float(row.get("pairs_with_both_reid"), 0.0) or 0.0
    passing = safe_float(row.get("pairs_passing_reid_threshold"), 0.0) or 0.0
    selected = safe_float(row.get("selected_edges_before_conflict_filter"), 0.0) or 0.0
    merges = safe_float(row.get("merges_applied"), None)
    if merges is None:
        merges = safe_float(row.get("applied_merge_mapping_size"), 0.0) or 0.0
    return with_reid > 0.0 and (passing > 0.0 or selected > 0.0 or merges > 0.0)
