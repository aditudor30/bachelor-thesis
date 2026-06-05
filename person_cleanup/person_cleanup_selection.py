"""Selection logic for Person cleanup runs."""

from typing import Any, Dict, List, Optional

from deep_oc_sort_3d.person_cleanup.person_cleanup_metrics import compute_cleanup_deltas
from deep_oc_sort_3d.person_cleanup.person_cleanup_io import safe_float


DEFAULT_CRITERIA = {
    "require_track1_validation_errors_zero": True,
    "require_non_person_unchanged": True,
    "max_purity_drop_vs_v2_current": 0.003,
    "max_false_merge_increase_vs_v2_current": 0.005,
    "min_person_fragmentation_reduction": 0.10,
    "prefer_track1_rows_reduction": True,
}


def select_best_person_cleanup(
    runs: List[Dict[str, Any]],
    v2_current: Dict[str, Any],
    criteria: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Select best Person cleanup run."""
    cfg = dict(DEFAULT_CRITERIA)
    if criteria:
        cfg.update(criteria)
    evaluated = []
    for run in runs:
        row = dict(run)
        row.update(compute_cleanup_deltas(row, v2_current, "vs_v2"))
        row["accepted_by_selection_criteria"] = is_acceptable(row, cfg)
        row["selection_score"] = score_run(row, cfg)
        evaluated.append(row)
    candidates = [row for row in evaluated if row["accepted_by_selection_criteria"]]
    if not candidates:
        if not evaluated:
            return {"verdict": "person_cleanup_not_beneficial", "best_run": None, "runs": []}
        best = sorted(evaluated, key=lambda row: float(row.get("selection_score", -1e9)), reverse=True)[0]
        verdict = "person_cleanup_reduces_rows_but_may_drop_valid_tracks"
        if (safe_float(best.get("vs_v2_track1_rows_delta"), 0.0) or 0.0) >= 0.0:
            verdict = "person_cleanup_not_beneficial"
        return {"verdict": verdict, "best_run": best.get("run_name"), "runs": evaluated}
    best = sorted(candidates, key=lambda row: float(row.get("selection_score", -1e9)), reverse=True)[0]
    verdict = "person_cleanup_ready_for_submission_candidate"
    if (safe_float(best.get("vs_v2_person_fragmentation_reduction"), 0.0) or 0.0) < float(cfg["min_person_fragmentation_reduction"]):
        verdict = "person_cleanup_valid_but_conservative"
    return {"verdict": verdict, "best_run": best.get("run_name"), "runs": evaluated, "criteria": cfg}


def is_acceptable(row: Dict[str, Any], criteria: Dict[str, Any]) -> bool:
    """Return True if a cleanup run satisfies hard criteria."""
    if bool(criteria.get("require_track1_validation_errors_zero", True)):
        if safe_float(row.get("track1_validation_errors"), None) != 0:
            return False
    if bool(criteria.get("require_non_person_unchanged", True)):
        non_person_delta = safe_float(row.get("vs_v2_non_person_rows_delta"), None)
        if non_person_delta is None or abs(non_person_delta) > float(criteria.get("max_non_person_rows_delta", 0)):
            return False
    purity_delta = safe_float(row.get("vs_v2_global_purity_mean_delta"), None)
    if purity_delta is not None and purity_delta < -float(criteria.get("max_purity_drop_vs_v2_current", 0.003)):
        return False
    false_merge_delta = safe_float(row.get("vs_v2_false_merge_rate_delta"), None)
    if false_merge_delta is not None and false_merge_delta > float(criteria.get("max_false_merge_increase_vs_v2_current", 0.005)):
        return False
    rows_delta = safe_float(row.get("vs_v2_track1_rows_delta"), None)
    if bool(criteria.get("prefer_track1_rows_reduction", True)) and rows_delta is not None and rows_delta >= 0:
        return False
    person_frag_delta = safe_float(row.get("vs_v2_person_fragmentation_approx_delta"), None)
    if person_frag_delta is not None and person_frag_delta >= 0:
        return False
    return True


def score_run(row: Dict[str, Any], criteria: Dict[str, Any]) -> float:
    """Score run as a row reduction / risk trade-off."""
    row_reduction = -(safe_float(row.get("vs_v2_track1_rows_delta"), 0.0) or 0.0)
    person_row_reduction = -(safe_float(row.get("vs_v2_person_rows_delta"), 0.0) or 0.0)
    non_person_penalty = abs(safe_float(row.get("vs_v2_non_person_rows_delta"), 0.0) or 0.0) * 10.0
    purity_penalty = max(0.0, -(safe_float(row.get("vs_v2_global_purity_mean_delta"), 0.0) or 0.0)) * 10000.0
    false_merge_penalty = max(0.0, safe_float(row.get("vs_v2_false_merge_rate_delta"), 0.0) or 0.0) * 10000.0
    validation_penalty = 0.0 if safe_float(row.get("track1_validation_errors"), None) == 0 else 1e9
    return float(row_reduction) + 0.25 * float(person_row_reduction) - non_person_penalty - purity_penalty - false_merge_penalty - validation_penalty

