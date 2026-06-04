"""Markdown report builder for the baseline 3D audit."""

from typing import Any, Dict


def build_3d_audit_report(
    track1_summary: Dict[str, Any],
    generic_summary: Dict[str, Any],
    smoothness_summary: Dict[str, Any],
    source_summary: Dict[str, Any],
    class_priors_summary: Dict[str, Any],
    projection_summary: Dict[str, Any],
) -> str:
    """Build the final Track1 3D audit Markdown report."""
    lines = [
        "# Track1 3D Field Audit",
        "",
        "## Executive Summary",
        "",
        _executive_summary(track1_summary, smoothness_summary, source_summary, class_priors_summary, projection_summary),
        "",
        "## Structural And Numeric Validity",
        "",
        "- Track1 rows audited: %s" % track1_summary.get("row_count", 0),
        "- Invalid Track1 column-count rows: %s" % track1_summary.get("invalid_row_count", 0),
        "- Generic export rows audited: %s" % generic_summary.get("row_count", 0),
        "- Generic vs Track1 dedup difference: %s"
        % generic_summary.get("generic_vs_track1", {}).get("dedup_difference_rows", "unknown"),
        "",
        "## 3D Field Plausibility",
        "",
        _field_plausibility(track1_summary),
        "",
        "## Source Provenance",
        "",
        "- Records with unknown 3D source: %s" % source_summary.get("records_with_unknown_source", 0),
        "- Records with explicit source metadata: %s" % source_summary.get("records_with_explicit_source_metadata", 0),
        "- Missing recommended metadata: %s" % ", ".join(source_summary.get("missing_recommended_metadata_fields", [])),
        "",
        str(source_summary.get("interpretation", "")),
        "",
        "## Smoothness",
        "",
        "- Objects audited: %s" % smoothness_summary.get("object_count", 0),
        "- Status distribution: `%s`" % smoothness_summary.get("status_distribution", {}),
        "- Max-step stats: `%s`" % _compact_stats(smoothness_summary.get("step_distance_max_stats", {})),
        "- Jump-count stats: `%s`" % _compact_stats(smoothness_summary.get("jump_count_stats", {})),
        "",
        "## Class Priors",
        "",
        "- Classes with priors: %s" % class_priors_summary.get("class_count", 0),
        "- Rows used for priors: %s" % class_priors_summary.get("row_count", 0),
        "",
        _class_prior_flags(class_priors_summary),
        "",
        "## Projection Audit",
        "",
        "- Records checked: %s" % projection_summary.get("total_records_checked", 0),
        "- Projection success: %s" % projection_summary.get("projection_success", 0),
        "- Projection failed: %s" % projection_summary.get("projection_failed", 0),
        "- Success rate: %s" % projection_summary.get("success_rate"),
        "- Failure reasons: `%s`" % projection_summary.get("failure_reasons", {}),
        "",
        "## MVP Assessment",
        "",
        _mvp_assessment(source_summary, smoothness_summary, class_priors_summary),
        "",
        "## Recommended Metadata",
        "",
        "- `source_3d`",
        "- `depth_source`",
        "- `dimensions_source`",
        "- `yaw_source`",
        "- `is_gt_derived`",
        "- `is_estimated_for_test`",
        "",
        "## Next Step: Pasul 15B",
        "",
        "Use the class-wise dimension priors and the failure modes from this audit to design a pseudo-3D estimator. "
        "That design should explicitly separate train/val GT diagnostics from test-time estimates and should write "
        "the metadata fields above into future Observation3D/frame-level records.",
        "",
    ]
    return "\n".join(lines)


def build_3d_audit_summary_json(
    track1_summary: Dict[str, Any],
    generic_summary: Dict[str, Any],
    smoothness_summary: Dict[str, Any],
    source_summary: Dict[str, Any],
    class_priors_summary: Dict[str, Any],
    projection_summary: Dict[str, Any],
) -> Dict[str, Any]:
    """Build a compact machine-readable final summary."""
    return {
        "track1_rows": track1_summary.get("row_count", 0),
        "track1_invalid_column_rows": track1_summary.get("invalid_row_count", 0),
        "generic_rows": generic_summary.get("row_count", 0),
        "dedup_difference_rows": generic_summary.get("generic_vs_track1", {}).get("dedup_difference_rows"),
        "smoothness_object_count": smoothness_summary.get("object_count", 0),
        "smoothness_status_distribution": smoothness_summary.get("status_distribution", {}),
        "source_counts": source_summary.get("source_counts", {}),
        "unknown_source_records": source_summary.get("records_with_unknown_source", 0),
        "class_prior_count": class_priors_summary.get("class_count", 0),
        "projection_success_rate": projection_summary.get("success_rate"),
        "projection_failure_reasons": projection_summary.get("failure_reasons", {}),
        "mvp_assessment": _mvp_assessment(source_summary, smoothness_summary, class_priors_summary),
    }


def _executive_summary(
    track1_summary: Dict[str, Any],
    smoothness_summary: Dict[str, Any],
    source_summary: Dict[str, Any],
    class_priors_summary: Dict[str, Any],
    projection_summary: Dict[str, Any],
) -> str:
    structural_ok = int(track1_summary.get("invalid_row_count", 0) or 0) == 0
    unknown_sources = int(source_summary.get("records_with_unknown_source", 0) or 0)
    source_rows = int(source_summary.get("record_count", 0) or 0)
    unknown_ratio = float(unknown_sources) / float(source_rows) if source_rows else 1.0
    if structural_ok and unknown_ratio > 0.5:
        return (
            "The current output appears structurally auditable, but much of the 3D provenance is not explicit. "
            "It is best described as a Track1-format MVP unless the numeric and projection checks below are clean "
            "and future runs record the 3D source metadata."
        )
    if structural_ok:
        return "The current output is structurally valid and has enough provenance to be assessed as a candidate 3D MTMC MVP."
    return "The current output has structural issues that should be fixed before using it as a 3D MTMC MVP."


def _field_plausibility(track1_summary: Dict[str, Any]) -> str:
    lines = []
    for field, stats in track1_summary.get("field_stats", {}).items():
        if not isinstance(stats, dict):
            continue
        lines.append(
            "- `%s`: valid=%s missing=%s nan=%s inf=%s min=%s max=%s median=%s"
            % (
                field,
                stats.get("valid_count"),
                stats.get("missing"),
                stats.get("nan"),
                stats.get("inf"),
                stats.get("min"),
                stats.get("max"),
                stats.get("median"),
            )
        )
    return "\n".join(lines) if lines else "No Track1 3D field stats were available."


def _class_prior_flags(class_priors_summary: Dict[str, Any]) -> str:
    flagged = []
    for class_key, item in class_priors_summary.get("classes", {}).items():
        if item.get("looks_constant_or_default"):
            flagged.append("class %s" % class_key)
    if not flagged:
        return "No class was flagged as obviously constant/default by the prior audit."
    return "Classes flagged as likely constant/default: %s." % ", ".join(flagged)


def _mvp_assessment(
    source_summary: Dict[str, Any],
    smoothness_summary: Dict[str, Any],
    class_priors_summary: Dict[str, Any],
) -> str:
    unknown = int(source_summary.get("records_with_unknown_source", 0) or 0)
    total = int(source_summary.get("record_count", 0) or 0)
    unknown_ratio = float(unknown) / float(total) if total else 1.0
    suspicious = int(smoothness_summary.get("status_distribution", {}).get("suspicious", 0) or 0)
    invalid = int(smoothness_summary.get("status_distribution", {}).get("invalid", 0) or 0)
    constant_classes = [
        key for key, value in class_priors_summary.get("classes", {}).items() if value.get("looks_constant_or_default")
    ]
    if unknown_ratio > 0.5:
        return (
            "Call this a Track1-format MVP, not yet a fully provenance-backed 3D MTMC MVP. "
            "The next version should write source metadata and replace unknown/default 3D fields with explicit estimates."
        )
    if invalid > 0 or suspicious > 0 or constant_classes:
        return (
            "This can be treated as a preliminary 3D MTMC MVP, but it has 3D stability/default-dimension risks that "
            "should guide Pasul 15B."
        )
    return "This is a reasonable baseline 3D MTMC MVP candidate, subject to external benchmark validation."


def _compact_stats(stats: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "valid_count": stats.get("valid_count"),
        "mean": stats.get("mean"),
        "median": stats.get("median"),
        "p95": stats.get("p95"),
        "max": stats.get("max"),
    }
