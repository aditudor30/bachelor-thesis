"""Comparison helpers for the pseudo-3D 6-camera experiment."""

from typing import Any, Dict, List


def compare_metric_dicts(v1: Dict[str, Any], v2: Dict[str, Any]) -> Dict[str, Any]:
    """Build a compact comparison from V1 and V2 metric dictionaries."""
    return {
        "v1": v1,
        "v2": v2,
        "deltas": compute_metric_deltas(v1, v2),
        "verdict": build_sixcam_verdict({"v1": v1, "v2": v2}),
    }


def compute_metric_deltas(v1: Dict[str, Any], v2: Dict[str, Any]) -> Dict[str, Any]:
    """Compute high-level V2 minus V1 deltas for comparable metrics."""
    return {
        "observations_delta": _delta(_section_value(v1, "observations", "num_observations"), _section_value(v2, "observations", "num_observations")),
        "local_records_delta": _delta(_section_value(v1, "local_tracking", "num_records"), _section_value(v2, "local_tracking", "num_records")),
        "active_tracks_delta": _delta(_section_value(v1, "local_tracking", "active_tracks"), _section_value(v2, "local_tracking", "active_tracks")),
        "tracklets_delta": _delta(_section_value(v1, "tracklets", "total_tracklets"), _section_value(v2, "tracklets", "total_tracklets")),
        "valid_tracklets_delta": _delta(_section_value(v1, "tracklets", "valid_tracklets"), _section_value(v2, "tracklets", "valid_tracklets")),
        "candidates_delta": _delta(_section_value(v1, "candidates", "total_candidates"), _section_value(v2, "candidates", "total_candidates")),
        "motion_invalid_delta": _delta(_section_value(v1, "motion_clean", "motion_invalid"), _section_value(v2, "motion_clean", "motion_invalid")),
        "final_rows_delta": _delta(_section_value(v1, "final_export", "rows"), _section_value(v2, "final_export", "rows")),
        "v2_pseudo3d_used_rate": _section_value(v2, "observations", "pseudo3d_used_rate"),
    }


def build_sixcam_verdict(summary: Dict[str, Any]) -> Dict[str, Any]:
    """Choose a conservative verdict for the 6-camera experiment."""
    v2 = summary.get("v2", {})
    pseudo_rate = _section_value(v2, "observations", "pseudo3d_used_rate")
    final_rows_delta = compute_metric_deltas(summary.get("v1", {}), v2).get("final_rows_delta")
    if pseudo_rate is None:
        label = "improve_pseudo3d_before_extension"
        reason = "pseudo3d coverage metric is unavailable"
    elif float(pseudo_rate) < 0.94:
        label = "improve_pseudo3d_before_extension"
        reason = "pseudo3d_used_rate is below 0.94"
    elif final_rows_delta is not None and float(final_rows_delta) > 0.0:
        label = "keep_v1_for_submission_use_v2_for_3d_analysis"
        reason = "pseudo3D coverage is high, but V2 produces more final rows than V1"
    else:
        label = "extend_pseudo3d_to_all_cameras"
        reason = "pseudo3D coverage is high and no obvious final export expansion was detected"
    return {"label": label, "reason": reason, "pseudo3d_used_rate": pseudo_rate}


def metric_delta_rows(deltas: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Convert deltas to CSV rows."""
    return [{"metric": key, "delta": value} for key, value in sorted(deltas.items())]


def _section_value(metrics: Dict[str, Any], section: str, key: str) -> Any:
    data = metrics.get(section, {})
    if not isinstance(data, dict):
        return None
    return data.get(key)


def _delta(a: Any, b: Any) -> Any:
    if a is None or b is None:
        return None
    try:
        return float(b) - float(a)
    except (TypeError, ValueError):
        return None

