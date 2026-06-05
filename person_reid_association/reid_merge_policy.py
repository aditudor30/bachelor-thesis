"""Merge policy for conservative ReID-guided Person association."""

from typing import Any, Dict, List, Tuple

from deep_oc_sort_3d.person_association.person_merge_policy import (
    mapping_from_edges,
    mapping_rows,
    remove_conflicting_mappings,
    summarize_merge_audit,
)
from deep_oc_sort_3d.person_reid_association.reid_association_io import TrackKey, safe_float


def build_reid_person_merge_mapping(
    scored_pairs: List[Dict[str, Any]],
    all_rows: List[Dict[str, Any]],
    config: Dict[str, Any],
) -> Tuple[Dict[TrackKey, str], List[Dict[str, Any]], Dict[str, Any]]:
    """Build Person-only merge mapping from ReID-scored pairs."""
    edges, audit = select_reid_merge_edges(scored_pairs, config)
    if not bool(config.get("apply_merges", True)):
        mapping: Dict[TrackKey, str] = {}
        summary = summarize_reid_merge_audit(audit, mapping)
        return mapping, audit, summary
    mapping = mapping_from_edges(edges)
    conflict_rows = []
    if bool(config.get("prevent_duplicate_frame_keys", True)):
        mapping, conflict_rows = remove_conflicting_mappings(all_rows, mapping)
        audit.extend(conflict_rows)
    summary = summarize_reid_merge_audit(audit, mapping)
    return mapping, audit, summary


def select_reid_merge_edges(scored_pairs: List[Dict[str, Any]], config: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Select merge edges using ReID threshold and safety gates."""
    threshold = config.get("reid_similarity_threshold")
    min_conf = float(config.get("min_mean_confidence", 0.03))
    reject_known_false = bool(config.get("reject_known_false_gt", True))
    max_combined = config.get("max_combined_pair_score")
    edges = []
    audit = []
    for row in scored_pairs:
        decision = dict(row)
        reason = "ok"
        if str(row.get("candidate_status", "ok")) != "ok":
            reason = str(row.get("reject_reason", "candidate_rejected"))
        elif str(row.get("reid_status")) != "ok":
            reason = "missing_reid"
        elif threshold is not None and (safe_float(row.get("reid_similarity"), -1.0) or -1.0) < float(threshold):
            reason = "reid_similarity_below_threshold"
        elif (safe_float(row.get("min_mean_confidence"), 0.0) or 0.0) < min_conf:
            reason = "confidence_too_low"
        elif max_combined is not None and (safe_float(row.get("combined_pair_score"), 1.0) or 1.0) > float(max_combined):
            reason = "combined_score_too_high"
        elif reject_known_false and row.get("reid_gt_pair_label") == "different_gt":
            reason = "known_false_gt_diagnostic"
        elif reject_known_false and row.get("same_gt_diagnostic") == "false_match":
            reason = "known_false_gt_diagnostic"
        decision["merge_selected"] = reason == "ok"
        decision["merge_reject_reason"] = reason
        audit.append(decision)
        if reason == "ok":
            edges.append(decision)
    return edges, audit


def summarize_reid_merge_audit(audit_rows: List[Dict[str, Any]], mapping: Dict[TrackKey, str]) -> Dict[str, Any]:
    """Summarize ReID merge audit rows."""
    base = summarize_merge_audit(audit_rows, mapping)
    selected = [row for row in audit_rows if row.get("merge_selected") in (True, "True", "true", "1")]
    base.update(
        {
            "selected_edges_with_reid": len([row for row in selected if row.get("reid_status") == "ok"]),
            "selected_same_gt_diagnostic": len([row for row in selected if row.get("reid_gt_pair_label") == "same_gt"]),
            "selected_different_gt_diagnostic": len([row for row in selected if row.get("reid_gt_pair_label") == "different_gt"]),
            "selected_unknown_gt_diagnostic": len([row for row in selected if row.get("reid_gt_pair_label") not in ("same_gt", "different_gt")]),
            "selected_reid_similarity_mean": _mean([row.get("reid_similarity") for row in selected]),
        }
    )
    return base


def _mean(values: List[Any]) -> Any:
    numeric = [safe_float(value, None) for value in values]
    numeric = [value for value in numeric if value is not None]
    if not numeric:
        return None
    return float(sum(numeric)) / float(len(numeric))
