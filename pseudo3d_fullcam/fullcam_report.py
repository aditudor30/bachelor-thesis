"""Markdown report builder for Step 15G full-camera pseudo-3D generation."""

from typing import Any, Dict, List


def build_fullcam_generation_report(summary: Dict[str, Any]) -> str:
    """Build the Step 15G full-camera pseudo-3D Markdown report."""
    lines = [
        "# Pseudo3D Full-Camera Generation Report",
        "",
        "## Scope",
        "",
        "Step 15G generates raw and stabilized pseudo3D predictions for every camera file required by the current pipeline inputs. It does not rerun local tracking, global association, or Track1 export.",
        "",
        "## Coverage",
        "",
        "- Required camera files: %s" % _value(summary.get("required_camera_files")),
        "- Raw prediction files: %s" % _value(summary.get("raw_files_existing")),
        "- Stabilized prediction files: %s" % _value(summary.get("stabilized_files_existing")),
        "- Raw file coverage: %s" % _value(summary.get("raw_file_coverage")),
        "- Stabilized file coverage: %s" % _value(summary.get("stabilized_file_coverage")),
        "- Expected records: %s" % _value(summary.get("total_records_expected")),
        "- Raw predictions: %s" % _value(summary.get("total_raw_predictions")),
        "- Stabilized predictions: %s" % _value(summary.get("total_stabilized_predictions")),
        "",
        "## Prediction Quality",
        "",
        "- Raw success rate: %s" % _value(summary.get("success_rate_raw")),
        "- Stabilized success rate: %s" % _value(summary.get("success_rate_stabilized")),
        "- Projection success rate: %s" % _value(summary.get("projection", {}).get("projection_success_rate")),
        "- Failed camera count: %s" % _value(summary.get("failed_camera_count")),
        "",
        "## Source Metadata",
        "",
    ]
    metadata = summary.get("source_metadata_completeness", {})
    if isinstance(metadata, dict) and metadata:
        for key in sorted(metadata.keys()):
            lines.append("- %s: %s" % (key, _value(metadata.get(key))))
    else:
        lines.append("- not available")
    lines.extend(
        [
            "",
            "## Smoothness",
            "",
        ]
    )
    smoothness = summary.get("smoothness", {})
    if isinstance(smoothness, dict) and smoothness:
        lines.extend(_dict_lines(smoothness))
    else:
        lines.append("- not available")
    lines.extend(
        [
            "",
            "## Recommendation",
            "",
            "- Recommendation: %s" % _value(summary.get("recommendation")),
            "- Can continue to Step 15H: %s" % _value(summary.get("can_continue_to_15h")),
            "",
            "Step 15H should use:",
            "",
            "`output/pseudo3d/baseline_v2_pseudo3d_fullcam/predictions_stabilized/`",
        ]
    )
    return "\n".join(lines) + "\n"


def _dict_lines(data: Dict[str, Any]) -> List[str]:
    rows = []
    for key in sorted(data.keys()):
        value = data.get(key)
        if isinstance(value, dict):
            rows.append("- %s: %s" % (key, _value(value)))
        else:
            rows.append("- %s: %s" % (key, _value(value)))
    return rows


def _value(value: Any) -> str:
    if value is None:
        return "None"
    return str(value)
