"""Markdown report generation for global tuning."""

from pathlib import Path
from typing import Any, Dict, List


def write_global_tuning_report(comparison: Dict[str, Any], output_path: Path) -> None:
    """Write a concise Markdown report for tuning results."""
    lines = build_global_tuning_report(comparison)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_global_tuning_report(comparison: Dict[str, Any]) -> List[str]:
    """Build report lines."""
    runs = comparison.get("runs", [])
    recommendation = comparison.get("best_run_recommendation", {})
    lines = [
        "# Global Association Tuning Report",
        "",
        "## Executive Summary",
        "",
        "This report compares controlled global association presets for `baseline_v2_pseudo3d_fullcam`.",
        "The sweep starts from motion-clean V2 full-camera candidates and reruns only downstream stages.",
        "",
        "Best run: `%s`" % recommendation.get("best_run"),
        "Verdict: `%s`" % recommendation.get("verdict"),
        "",
        "## Recap",
        "",
        "Step 15H improved pseudo3D coverage and multi-camera output but increased diagnostic fragmentation.",
        "Step 15I attributed the dominant issue to global association rather than detector, pseudo3D, local tracking, or motion filtering.",
        "",
        "## Run Metrics",
        "",
        "| run | frag | purity | false merge | Track1 rows | MC tracks | accepted edges | transition edges |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in runs:
        lines.append(
            "| %s | %s | %s | %s | %s | %s | %s | %s |"
            % (
                row.get("run_name"),
                _fmt(row.get("fragmentation_approx")),
                _fmt(row.get("global_purity_mean")),
                _fmt(row.get("false_merge_rate")),
                _fmt(row.get("track1_rows")),
                _fmt(row.get("multi_camera_tracks")),
                _fmt(row.get("accepted_edges")),
                _fmt(row.get("transition_edges_accepted")),
            )
        )
    lines.extend(
        [
            "",
            "## Deltas vs V2 Current",
            "",
            "| run | frag reduction | purity delta | false merge delta | Track1 rows delta |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in runs:
        lines.append(
            "| %s | %s | %s | %s | %s |"
            % (
                row.get("run_name"),
                _fmt(row.get("vs_v2_fragmentation_reduction")),
                _fmt(row.get("vs_v2_global_purity_mean_delta")),
                _fmt(row.get("vs_v2_false_merge_rate_delta")),
                _fmt(row.get("vs_v2_track1_rows_delta")),
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Lower fragmentation is useful only if purity and false merge rate remain controlled.",
            "- `conservative_compact` is an export-policy diagnostic, not a change to the core Track1 writer.",
            "- GT is used only for diagnostic metrics on validation/holdout subsets; test remains GT-free.",
            "",
            "## Next Step",
            "",
            "Step 15K should promote the selected preset only if Track1 validation stays clean and the trade-off is better than V2 current.",
        ]
    )
    return lines


def _fmt(value: Any) -> str:
    if value is None:
        return "not_available"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if abs(number) >= 1000:
        return "%.0f" % number
    return "%.6f" % number

