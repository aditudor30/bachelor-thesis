"""Metrics and final verdicts for Person ReID visual decision."""

from typing import Any, Dict, List

from deep_oc_sort_3d.reid_visual_decision.visual_decision_io import safe_float


def summarize_visual_rows(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Summarize auto labels and risk for a list of reviewed events."""
    counts: Dict[str, int] = {}
    by_variant: Dict[str, Dict[str, int]] = {}
    risks = []
    for row in rows:
        label = str(row.get("auto_label", "unknown"))
        variant = str(row.get("variant", "unknown"))
        counts[label] = counts.get(label, 0) + 1
        by_variant.setdefault(variant, {})
        by_variant[variant][label] = by_variant[variant].get(label, 0) + 1
        risk = safe_float(row.get("risk_score"), None)
        if risk is not None:
            risks.append(risk)
    return {
        "total_review_events": len(rows),
        "auto_label_counts": counts,
        "auto_label_counts_by_variant": by_variant,
        "mean_risk_score": float(sum(risks)) / float(len(risks)) if risks else None,
        "likely_good_count": counts.get("likely_good", 0),
        "suspicious_or_bad_count": counts.get("suspicious", 0) + counts.get("likely_bad", 0),
        "not_enough_visual_evidence_count": counts.get("not_enough_visual_evidence", 0),
    }


def decide_final_variant(summary: Dict[str, Any], selected_variant: Dict[str, Any]) -> Dict[str, Any]:
    """Choose a conservative visual-decision verdict."""
    counts = summary.get("auto_label_counts", {})
    total = int(summary.get("total_review_events", 0) or 0)
    bad = int(counts.get("likely_bad", 0) or 0)
    suspicious = int(counts.get("suspicious", 0) or 0)
    no_evidence = int(counts.get("not_enough_visual_evidence", 0) or 0)
    likely_good = int(counts.get("likely_good", 0) or 0)
    selected_name = str(selected_variant.get("best_run") or selected_variant.get("selected_variant") or "combined_safe_080")
    if total <= 0:
        verdict = "finetuned_reid_visuals_invalid_fix_required"
        reason = "no_visual_events_reviewed"
    elif no_evidence > total * 0.50:
        verdict = "finetuned_reid_visuals_invalid_fix_required"
        reason = "too_many_events_without_visual_evidence"
    elif bad > 0 or suspicious > likely_good:
        verdict = "finetuned_reid_visuals_too_ambiguous"
        reason = "suspicious_or_bad_events_not_clearly_outnumbered"
    elif selected_name == "combined_safe_080":
        verdict = "combined_safe_080_keep_as_experimental_final"
        reason = "visuals_do_not_contradict_selected_safe_variant"
    elif selected_name == "threshold_080":
        verdict = "threshold_080_keep_as_reid_diagnostic_only"
        reason = "pure_reid_gain_is_small"
    else:
        verdict = "finetuned_reid_visuals_promising_but_not_final"
        reason = "selected_variant_not_one_of_primary_targets"
    return {
        "final_verdict": verdict,
        "reason": reason,
        "selected_variant_from_step18c": selected_name,
        "total_review_events": total,
        "likely_good_count": likely_good,
        "suspicious_count": suspicious,
        "likely_bad_count": bad,
        "not_enough_visual_evidence_count": no_evidence,
    }

