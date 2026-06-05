"""Best-run selection logic for global tuning sweeps."""

from typing import Any, Dict, List, Optional

from deep_oc_sort_3d.global_tuning.tuning_io import safe_float
from deep_oc_sort_3d.global_tuning.tuning_metrics import compute_metric_deltas


DEFAULT_SELECTION_CRITERIA = {
    "max_purity_drop_vs_v2_current": 0.005,
    "max_false_merge_increase_vs_v2_current": 0.01,
    "min_fragmentation_reduction": 0.10,
    "prefer_track1_rows_reduction": True,
    "require_track1_validation_errors_zero": True,
}


def select_best_run(
    run_metrics: List[Dict[str, Any]],
    v2_current: Dict[str, Any],
    criteria: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Select the best run under fixed trade-off criteria."""
    cfg = dict(DEFAULT_SELECTION_CRITERIA)
    if criteria:
        cfg.update(criteria)
    evaluated = []
    for metrics in run_metrics:
        item = dict(metrics)
        item.update(compute_metric_deltas(metrics, v2_current, "vs_v2"))
        item["accepted_by_selection_criteria"] = is_acceptable_run(item, cfg)
        item["selection_score"] = selection_score(item, cfg)
        evaluated.append(item)
    acceptable = [item for item in evaluated if item["accepted_by_selection_criteria"]]
    candidates = acceptable if acceptable else evaluated
    if not candidates:
        return {
            "verdict": "no_tuning_run_improved_tradeoff",
            "best_run": None,
            "reason": "no_runs",
            "runs": [],
        }
    best = sorted(candidates, key=lambda item: float(item.get("selection_score", -1e9)), reverse=True)[0]
    verdict = "best_run_ready_for_submission_candidate" if best["accepted_by_selection_criteria"] else "tuning_needs_second_round"
    if not best["accepted_by_selection_criteria"]:
        verdict = "best_run_valid_but_needs_more_tuning" if _has_zero_validation_errors(best) else "tuning_needs_second_round"
    return {
        "verdict": verdict,
        "best_run": best.get("run_name"),
        "selection_score": best.get("selection_score"),
        "accepted_by_selection_criteria": best.get("accepted_by_selection_criteria"),
        "criteria": cfg,
        "runs": evaluated,
    }


def is_acceptable_run(metrics: Dict[str, Any], criteria: Dict[str, Any]) -> bool:
    """Return True when a run satisfies all hard selection criteria."""
    if bool(criteria.get("require_track1_validation_errors_zero", True)) and not _has_zero_validation_errors(metrics):
        return False
    purity_delta = safe_float(metrics.get("vs_v2_global_purity_mean_delta"), None)
    if purity_delta is not None and purity_delta < -float(criteria.get("max_purity_drop_vs_v2_current", 0.005)):
        return False
    false_merge_delta = safe_float(metrics.get("vs_v2_false_merge_rate_delta"), None)
    if false_merge_delta is not None and false_merge_delta > float(criteria.get("max_false_merge_increase_vs_v2_current", 0.01)):
        return False
    reduction = safe_float(metrics.get("vs_v2_fragmentation_reduction"), None)
    if reduction is None or reduction < float(criteria.get("min_fragmentation_reduction", 0.10)):
        return False
    return True


def selection_score(metrics: Dict[str, Any], criteria: Dict[str, Any]) -> float:
    """Compute a transparent scalar trade-off score."""
    reduction = safe_float(metrics.get("vs_v2_fragmentation_reduction"), 0.0) or 0.0
    purity_delta = safe_float(metrics.get("vs_v2_global_purity_mean_delta"), 0.0) or 0.0
    false_merge_delta = safe_float(metrics.get("vs_v2_false_merge_rate_delta"), 0.0) or 0.0
    row_delta = safe_float(metrics.get("vs_v2_track1_rows_delta"), 0.0) or 0.0
    row_bonus = 0.0
    if bool(criteria.get("prefer_track1_rows_reduction", True)):
        baseline_rows = safe_float(metrics.get("track1_rows"), None)
        if baseline_rows is not None and baseline_rows > 0:
            row_bonus = -row_delta / max(float(baseline_rows), 1.0)
    validation_penalty = 0.0 if _has_zero_validation_errors(metrics) else 10.0
    return float(reduction) + 0.5 * float(purity_delta) - 2.0 * max(0.0, float(false_merge_delta)) + 0.1 * row_bonus - validation_penalty


def _has_zero_validation_errors(metrics: Dict[str, Any]) -> bool:
    value = safe_float(metrics.get("track1_validation_errors"), None)
    return value == 0
