"""Root-cause heuristics for fragmentation audit outputs."""

from typing import Any, Dict, List


def analyze_root_cause(comparison: Dict[str, Any]) -> Dict[str, Any]:
    """Infer likely dominant fragmentation causes from metric deltas."""
    high = comparison.get("high_level", {})
    reasons = []
    scores = {
        "local_tracking_dominant": 0.0,
        "tracklet_filtering_dominant": 0.0,
        "motion_filtering_dominant": 0.0,
        "global_association_dominant": 0.0,
        "final_export_dominant": 0.0,
    }
    if _positive(high.get("local_track_delta")):
        scores["local_tracking_dominant"] += 2.0
        reasons.append("V2 increases local track count")
    if _positive(high.get("local_short_ratio_delta")):
        scores["local_tracking_dominant"] += 1.5
        reasons.append("V2 increases local short-track ratio")
    if _positive(high.get("tracklet_delta")):
        scores["tracklet_filtering_dominant"] += 1.0
        reasons.append("V2 increases tracklet count")
    if _positive(high.get("motion_invalid_ratio_delta")):
        scores["motion_filtering_dominant"] += 2.0
        reasons.append("V2 increases motion invalid ratio")
    if _positive(high.get("global_tracks_delta")):
        scores["global_association_dominant"] += 1.5
        reasons.append("V2 increases global track count")
    if _positive(high.get("global_singleton_ratio_delta")):
        scores["global_association_dominant"] += 2.0
        reasons.append("V2 increases global singleton ratio")
    if _positive(high.get("global_fragmentation_delta")):
        scores["global_association_dominant"] += 2.0
        reasons.append("V2 increases diagnostic global fragmentation")
    if _positive(high.get("generic_rows_delta")):
        scores["final_export_dominant"] += 1.5
        reasons.append("V2 writes more final export rows")
    max_score = max(scores.values()) if scores else 0.0
    dominant = [key for key, value in scores.items() if value == max_score and value > 0.0]
    verdict = dominant[0] if len(dominant) == 1 else "mixed_causes"
    return {
        "verdict": verdict,
        "scores": scores,
        "reasons": reasons,
        "tuning_recommendations": build_tuning_recommendations(verdict, scores),
    }


def build_tuning_recommendations(verdict: str, scores: Dict[str, float]) -> List[Dict[str, Any]]:
    """Return concrete recommendations for Step 15J, without applying them."""
    recommendations = []
    if verdict in ("local_tracking_dominant", "mixed_causes") or scores.get("local_tracking_dominant", 0.0) > 0.0:
        recommendations.append(
            {
                "area": "local_tracking",
                "action": "Run an ablation with higher max_misses and class-specific max_3d_distance.",
                "metrics_to_watch": "local short ratio, local track count, GT fragmentation on val/holdout",
            }
        )
    if verdict in ("tracklet_filtering_dominant", "mixed_causes") or scores.get("tracklet_filtering_dominant", 0.0) > 0.0:
        recommendations.append(
            {
                "area": "tracklet_filtering",
                "action": "Audit min-length rules and consider merging or suppressing very short tracklets.",
                "metrics_to_watch": "valid tracklets, short ratio, candidate count",
            }
        )
    if verdict in ("motion_filtering_dominant", "mixed_causes") or scores.get("motion_filtering_dominant", 0.0) > 0.0:
        recommendations.append(
            {
                "area": "motion_filtering",
                "action": "Use thresholds calibrated for stabilized pseudo3D rather than V1 geometry.",
                "metrics_to_watch": "motion_invalid ratio, p95/p99 step distance, jump ratio",
            }
        )
    if verdict in ("global_association_dominant", "mixed_causes") or scores.get("global_association_dominant", 0.0) > 0.0:
        recommendations.append(
            {
                "area": "global_association",
                "action": "Relax transition/overlap association thresholds in a controlled sweep.",
                "metrics_to_watch": "multi-camera tracks, purity, false merge rate, fragmentation",
            }
        )
    if verdict in ("final_export_dominant", "mixed_causes") or scores.get("final_export_dominant", 0.0) > 0.0:
        recommendations.append(
            {
                "area": "final_export",
                "action": "Evaluate a compact export policy for very short low-confidence global tracks.",
                "metrics_to_watch": "Track1 rows, rows per track p95, validation errors",
            }
        )
    return recommendations


def _positive(value: Any) -> bool:
    try:
        return float(value) > 0.0
    except (TypeError, ValueError):
        return False

