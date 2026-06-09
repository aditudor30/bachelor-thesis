"""Heuristic visual-risk classifier for Person ReID merge events."""

from typing import Any, Dict

from deep_oc_sort_3d.reid_visual_decision.visual_decision_io import safe_float, safe_int


LABELS = ["likely_good", "ambiguous", "suspicious", "likely_bad", "not_enough_visual_evidence"]


def classify_merge_event(event: Dict[str, Any], evidence: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    """Classify a merge event using ReID, geometry, and visual evidence counts."""
    heuristics = config.get("heuristics", {}) if isinstance(config, dict) else {}
    min_crops = int(heuristics.get("min_crops_per_fragment", 2))
    sim_threshold = float(heuristics.get("reid_similarity_threshold", 0.80))
    high_sim = float(heuristics.get("high_similarity", 0.86))
    suspicious_gap = float(heuristics.get("suspicious_temporal_gap", 220.0))
    suspicious_dist = float(heuristics.get("suspicious_spatial_distance", 12.0))
    bad_dist = float(heuristics.get("bad_spatial_distance", 20.0))
    crops_a = int(evidence.get("num_crops_a", 0) or 0)
    crops_b = int(evidence.get("num_crops_b", 0) or 0)
    sim = safe_float(event.get("reid_similarity"), None)
    gap = abs(safe_float(event.get("temporal_gap"), 0.0) or 0.0)
    dist = safe_float(event.get("spatial_distance"), 0.0) or 0.0
    reasons = []
    if crops_a < min_crops or crops_b < min_crops:
        return {
            "auto_label": "not_enough_visual_evidence",
            "risk_score": 1.0,
            "risk_reasons": "insufficient_crops:%d/%d" % (crops_a, crops_b),
        }
    if str(event.get("reid_gt_pair_label")) == "different_gt" or str(event.get("same_gt_diagnostic")) == "false_match":
        reasons.append("known_false_gt_diagnostic")
        return {"auto_label": "likely_bad", "risk_score": 0.95, "risk_reasons": ";".join(reasons)}
    if sim is None:
        reasons.append("missing_reid_similarity")
        return {"auto_label": "suspicious", "risk_score": 0.75, "risk_reasons": ";".join(reasons)}
    risk = 0.0
    if sim < sim_threshold:
        risk += 0.45
        reasons.append("similarity_below_threshold")
    elif sim < sim_threshold + 0.03:
        risk += 0.20
        reasons.append("near_threshold")
    if dist >= bad_dist:
        risk += 0.45
        reasons.append("very_large_spatial_distance")
    elif dist >= suspicious_dist:
        risk += 0.25
        reasons.append("large_spatial_distance")
    if gap >= suspicious_gap:
        risk += 0.20
        reasons.append("large_temporal_gap")
    if sim >= high_sim and dist < suspicious_dist and gap < suspicious_gap:
        reasons.append("high_similarity_geometry_ok")
        return {"auto_label": "likely_good", "risk_score": min(risk, 0.20), "risk_reasons": ";".join(reasons)}
    if risk >= 0.70:
        label = "likely_bad"
    elif risk >= 0.35:
        label = "suspicious"
    else:
        label = "ambiguous"
    if not reasons:
        reasons.append("moderate_similarity_or_geometry")
    return {"auto_label": label, "risk_score": risk, "risk_reasons": ";".join(reasons)}

