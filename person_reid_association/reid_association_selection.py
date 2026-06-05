"""Selection logic for ReID-guided Person association runs."""

from typing import Any, Dict, List, Optional

from deep_oc_sort_3d.person_reid_association.reid_association_metrics import compute_reid_association_deltas


DEFAULT_SELECTION = {
    "require_track1_errors_zero": True,
    "require_non_person_unchanged": True,
    "max_false_merge_increase": 0.005,
    "max_purity_drop": 0.003,
    "require_non_noop": True,
}


def select_best_reid_association(
    runs: List[Dict[str, Any]],
    v2_current: Dict[str, Any],
    criteria: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Select best ReID-guided association run."""
    cfg = dict(DEFAULT_SELECTION)
    if criteria:
        cfg.update(criteria)
    evaluated = []
    for run in runs:
        row = dict(run)
        row.update(compute_reid_association_deltas(row, v2_current, "vs_v2"))
        row["accepted_by_selection_criteria"] = _acceptable(row, cfg)
        row["selection_score"] = _score(row)
        evaluated.append(row)
    accepted = [row for row in evaluated if row.get("accepted_by_selection_criteria")]
    if not accepted:
        if any(float(row.get("merges_applied") or 0) > 0 for row in evaluated):
            verdict = "reid_association_too_risky"
        else:
            verdict = "needs_better_reid_or_domain_tuning"
        return {"verdict": verdict, "best_run": None, "runs": evaluated, "criteria": cfg}
    best = sorted(accepted, key=lambda row: float(row.get("selection_score", -1e9)), reverse=True)[0]
    frag_delta = best.get("vs_v2_person_fragmentation_approx_delta")
    if frag_delta is None or float(frag_delta) >= 0:
        verdict = "reid_association_valid_but_minor_gain"
    else:
        verdict = "reid_association_improves_v2"
    return {"verdict": verdict, "best_run": best.get("run_name"), "runs": evaluated, "criteria": cfg}


def _acceptable(row: Dict[str, Any], criteria: Dict[str, Any]) -> bool:
    if row.get("run_status") not in (None, "ok"):
        return False
    if bool(criteria.get("require_non_noop", True)) and float(row.get("merges_applied") or 0) <= 0:
        return False
    if bool(criteria.get("require_track1_errors_zero", True)) and row.get("track1_validation_errors") not in (0, 0.0, "0"):
        return False
    if bool(criteria.get("require_non_person_unchanged", True)):
        delta = row.get("vs_v2_non_person_rows_delta")
        if delta is None or abs(float(delta)) > 0.0:
            return False
    purity_delta = row.get("vs_v2_global_purity_mean_delta")
    if purity_delta is not None and float(purity_delta) < -float(criteria.get("max_purity_drop", 0.003)):
        return False
    false_delta = row.get("vs_v2_false_merge_rate_delta")
    if false_delta is not None and float(false_delta) > float(criteria.get("max_false_merge_increase", 0.005)):
        return False
    frag_delta = row.get("vs_v2_person_fragmentation_approx_delta")
    if frag_delta is not None and float(frag_delta) >= 0:
        return False
    return True


def _score(row: Dict[str, Any]) -> float:
    frag_gain = -(float(row.get("vs_v2_person_fragmentation_approx_delta") or 0.0))
    purity_penalty = max(0.0, -(float(row.get("vs_v2_global_purity_mean_delta") or 0.0))) * 10000.0
    false_penalty = max(0.0, float(row.get("vs_v2_false_merge_rate_delta") or 0.0)) * 10000.0
    non_person_penalty = abs(float(row.get("vs_v2_non_person_rows_delta") or 0.0)) * 100.0
    return frag_gain + float(row.get("merges_applied") or 0.0) * 0.01 - purity_penalty - false_penalty - non_person_penalty

