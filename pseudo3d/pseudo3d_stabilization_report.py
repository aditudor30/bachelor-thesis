"""Markdown report builder for Step 15D pseudo-3D stabilization."""

from typing import Any, Dict


def build_stabilization_report(summary: Dict[str, Any]) -> str:
    """Build a concise Markdown report for stabilized pseudo-3D outputs."""
    lines = [
        "# Pseudo-3D Temporal Stabilization Report",
        "",
        "## Context",
        "",
        "Step 15D stabilizes the isolated Step 15C pseudo-3D predictions before any integration into Observation3D.",
        "The stabilizer uses only prediction-time fields: bbox-derived centers/depths, class priors, track ids, and frame order.",
        "Ground truth and depth maps are reserved for after-the-fact evaluation only.",
        "",
        "## Methods",
        "",
        "- Track grouping by subset, scene, camera, and local/global track id.",
        "- Median/EMA temporal smoothing for center and depth sequences.",
        "- Jump guard with configurable hold/interpolate/mark-invalid strategy.",
        "- Small bbox depth guard with confidence reduction.",
        "- Yaw recomputed from smoothed motion when displacement is sufficient.",
        "- Dimensions remain class-prior dimensions.",
        "",
        "## Summary",
        "",
        "- Predictions: %s" % summary.get("num_predictions"),
        "- Tracks: %s" % summary.get("num_tracks"),
        "- Success rate: %s" % summary.get("success_rate"),
        "- Center-smoothed records: %s" % summary.get("num_center_smoothed"),
        "- Depth-smoothed records: %s" % summary.get("num_depth_smoothed"),
        "- Jump-corrected records: %s" % summary.get("num_jump_corrected"),
        "- Small-bbox guarded records: %s" % summary.get("num_small_bbox_guarded"),
        "",
        "## Recommendation",
        "",
        _recommendation(summary),
        "",
    ]
    return "\n".join(lines)


def _recommendation(summary: Dict[str, Any]) -> str:
    invalid_rate = summary.get("stabilized_invalid_rate")
    if invalid_rate is None:
        return "Run smoothness, projection, and val/holdout evaluation before deciding on Step 15E integration."
    try:
        if float(invalid_rate) < 0.102:
            return "Stabilization improved the invalid smoothness rate versus Step 15C; proceed to Step 15E only after projection and metadata checks remain healthy."
    except (TypeError, ValueError):
        pass
    return "Do not integrate yet; tune smoothing and jump guard thresholds before Step 15E."
