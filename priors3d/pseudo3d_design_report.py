"""Generate Step 15B pseudo-3D design and roadmap documents."""

from typing import Any, Dict


def build_pseudo3d_estimator_design(config: Dict[str, Any]) -> str:
    """Build the pseudo-3D estimator design document."""
    primary = config.get("primary_method", "bbox_height_depth")
    fallbacks = config.get("fallback_methods", ["bottom_center_ground_approx", "class_default"])
    return "\n".join(
        [
            "# Pseudo-3D Estimator Design",
            "",
            "This is a Step 15B design document. It does not implement or activate the estimator.",
            "",
            "## Inputs",
            "",
            "- RGB frame",
            "- YOLO 2D bbox",
            "- `class_id` and class name",
            "- camera calibration",
            "- final class-wise 3D dimension priors",
            "- optional local/global track history",
            "",
            "## Forbidden Test-time Inputs",
            "",
            "- ground truth",
            "- depth maps",
            "- any GT-derived identity or 3D center",
            "",
            "## Output",
            "",
            "The estimator should output center, dimensions, yaw, depth, confidence, and full provenance metadata.",
            "",
            "## Method A: bbox-height depth estimation",
            "",
            "Use the real class height prior and the 2D bbox pixel height:",
            "",
            "```text",
            "depth ~= fy * real_height / bbox_pixel_height",
            "```",
            "",
            "Backproject the bbox center or bottom-center ray using the estimated depth. Attach dimensions from class priors.",
            "",
            "## Method B: bottom-center ground approximation",
            "",
            "Use the bbox bottom-center as an approximate ground-contact point. If calibration and a usable ground plane are available, intersect the camera ray with the ground plane. Otherwise fall back to Method A.",
            "",
            "## Method C: temporal smoothing and track refinement",
            "",
            "Smooth per-detection centers on each local track, reject large jumps, estimate yaw from motion direction when displacement is sufficient, and keep dimensions as class priors or track medians.",
            "",
            "## Method D: class-specific fallback",
            "",
            "When bboxes are too small, calibration is incomplete, or projection is unstable, keep class-prior dimensions, set low confidence, use default yaw, and mark center/depth as unknown or low-confidence estimated.",
            "",
            "## Fallback Hierarchy",
            "",
            "- Primary: `%s`" % primary,
            "- Fallbacks: `%s`" % "`, `".join([str(item) for item in fallbacks]),
            "",
            "## Failure Modes",
            "",
            "- tiny or truncated bbox",
            "- wrong class prior",
            "- calibration convention mismatch",
            "- non-grounded object contact point",
            "- temporal ID switch",
            "- yaw ambiguity for static objects",
            "",
            "## What Not To Claim",
            "",
            "Do not claim metric 3D accuracy on test without GT. Report this as a calibrated pseudo-3D estimate with explicit provenance and validation on val/holdout.",
            "",
        ]
    )


def build_pseudo3d_validation_plan() -> str:
    """Build the validation plan for baseline_v2_pseudo3d."""
    return "\n".join(
        [
            "# Pseudo-3D Validation Plan",
            "",
            "Evaluate pseudo-3D on official_val and internal_holdout as if depth and GT were unavailable at inference.",
            "",
            "## Allowed Inference Inputs",
            "",
            "- RGB",
            "- 2D bbox",
            "- class id",
            "- calibration",
            "- class priors",
            "",
            "## Forbidden Inference Inputs",
            "",
            "- GT",
            "- depth map",
            "",
            "GT and depth may be used only after inference for diagnostic targets.",
            "",
            "## Metrics",
            "",
            "- center error",
            "- depth error",
            "- dimension error",
            "- yaw error",
            "- projection success rate",
            "- trajectory smoothness",
            "- Track1 format validity",
            "- effect on global association",
            "",
            "## Minimum Success Criteria",
            "",
            "- source metadata complete",
            "- Track1 validation errors = 0",
            "- projection success comparable to or better than baseline_v1",
            "- smoothness invalid rate lower than baseline_v1",
            "- no major degradation in global association",
            "",
        ]
    )


def build_roadmap_baseline_v2() -> str:
    """Build the roadmap from Step 15B to baseline_v2_pseudo3d."""
    return "\n".join(
        [
            "# Roadmap: baseline_v2_pseudo3d",
            "",
            "## Step 15B",
            "",
            "Finalize class priors, source metadata schema, pseudo-3D design, config, and validation plan.",
            "",
            "## Step 15C",
            "",
            "Implement the pseudo-3D estimator as an isolated module and evaluate it on val/holdout as test.",
            "",
            "## Step 15D",
            "",
            "Integrate pseudo-3D outputs into Observation3D with explicit provenance fields.",
            "",
            "## Step 15E",
            "",
            "Re-run the full pipeline and export `baseline_v2_pseudo3d`.",
            "",
            "## Step 16",
            "",
            "Use the pseudo-3D baseline and diagnostics to scope a learned 3D head.",
            "",
        ]
    )


def build_step15b_summary(
    priors_summary: Dict[str, Any],
    comparison_summary: Dict[str, Any],
    schema: Dict[str, Any],
) -> Dict[str, Any]:
    """Build compact Step 15B summary JSON."""
    confidence_counts = {}
    fallback_count = 0
    for item in priors_summary.get("classes", []):
        level = str(item.get("confidence_level", "unknown"))
        confidence_counts[level] = confidence_counts.get(level, 0) + 1
        if item.get("fallback_required"):
            fallback_count += 1
    return {
        "class_count": priors_summary.get("class_count", 0),
        "confidence_counts": confidence_counts,
        "fallback_required_classes": fallback_count,
        "comparison_rows": comparison_summary.get("row_count", 0),
        "comparison_warning_counts": comparison_summary.get("warning_counts", {}),
        "metadata_schema_version": schema.get("version"),
        "next_step": "Step 15C: implement isolated pseudo-3D estimator and validate on val/holdout as test.",
    }


def build_step15b_summary_markdown(summary: Dict[str, Any]) -> str:
    """Build compact Step 15B summary Markdown."""
    return "\n".join(
        [
            "# Step 15B Summary",
            "",
            "- Classes with final priors: %s" % summary.get("class_count", 0),
            "- Confidence counts: `%s`" % summary.get("confidence_counts", {}),
            "- Fallback-required classes: %s" % summary.get("fallback_required_classes", 0),
            "- Comparison rows: %s" % summary.get("comparison_rows", 0),
            "- Comparison warnings: `%s`" % summary.get("comparison_warning_counts", {}),
            "- Metadata schema version: `%s`" % summary.get("metadata_schema_version"),
            "",
            "Next: %s" % summary.get("next_step"),
            "",
        ]
    )

