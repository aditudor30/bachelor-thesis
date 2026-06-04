"""Report builder for Step 15C isolated pseudo-3D validation."""

from typing import Any, Dict


def build_pseudo3d_validation_report(
    extraction_summary: Dict[str, Any],
    eval_summary: Dict[str, Any],
    projection_summary: Dict[str, Any],
    smoothness_summary: Dict[str, Any],
) -> str:
    """Build Markdown report for isolated pseudo-3D validation."""
    lines = [
        "# Pseudo-3D Isolated Validation Report",
        "",
        "## Method",
        "",
        "The isolated Step 15C estimator uses bbox-height depth, calibration, class-prior dimensions, and default or motion yaw. It does not use GT or depth maps for prediction.",
        "",
        "## Extraction",
        "",
        "- Predictions: %s" % extraction_summary.get("num_predictions", 0),
        "- Success: %s" % extraction_summary.get("num_success", 0),
        "- Failed: %s" % extraction_summary.get("num_failed", 0),
        "- Success rate: %s" % extraction_summary.get("success_rate"),
        "- Failure reasons: `%s`" % extraction_summary.get("failure_reasons", {}),
        "",
        "## Source Metadata",
        "",
        "`%s`" % extraction_summary.get("source_metadata_completeness", {}),
        "",
        "## Evaluation",
        "",
        "- Evaluated predictions: %s" % eval_summary.get("num_evaluated", 0),
        "- Missing GT matches: %s" % eval_summary.get("num_missing_gt", 0),
        "- Center error: `%s`" % eval_summary.get("center_error", {}),
        "- Depth error: `%s`" % eval_summary.get("depth_error", {}),
        "- Dimension error: `%s`" % eval_summary.get("dimension_error", {}),
        "- Yaw error: `%s`" % eval_summary.get("yaw_error", {}),
        "",
        "## Projection",
        "",
        "- Projection success rate: %s" % projection_summary.get("projection_success_rate"),
        "- Projection failure reasons: `%s`" % projection_summary.get("failure_reasons", {}),
        "",
        "## Smoothness",
        "",
        "- Objects: %s" % smoothness_summary.get("object_count", 0),
        "- Status distribution: `%s`" % smoothness_summary.get("status_distribution", {}),
        "",
        "## Limitations",
        "",
        "- bbox-height depth is approximate",
        "- ground-plane estimation is only scaffolded unless an explicit ground plane is configured",
        "- GT/depth are not used for prediction, only for validation",
        "- camera/world coordinate validity depends on calibration conventions",
        "",
        "## Recommendation for Step 15D",
        "",
        "Integrate into Observation3D only if projection, smoothness, and validation errors are stable on official_val/internal_holdout. Otherwise tune Step 15C before integration.",
        "",
    ]
    return "\n".join(lines)

