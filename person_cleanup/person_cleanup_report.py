"""Markdown report for Person cleanup."""

from pathlib import Path
from typing import Any, Dict, List


def write_person_cleanup_report(summary: Dict[str, Any], output_path: Path) -> None:
    """Write Person cleanup report."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(build_person_cleanup_report(summary)) + "\n", encoding="utf-8")


def build_person_cleanup_report(summary: Dict[str, Any]) -> List[str]:
    """Build report lines."""
    recommendation = summary.get("best_person_cleanup_recommendation", {})
    runs = summary.get("runs", [])
    lines = [
        "# Person Cleanup Report",
        "",
        "## Executive Summary",
        "",
        "This report compares Person-specific cleanup experiments for `baseline_v2_pseudo3d_fullcam`.",
        "The experiments operate on separate final-export copies and do not modify baseline outputs.",
        "",
        "Best run: `%s`" % recommendation.get("best_run"),
        "Verdict: `%s`" % recommendation.get("verdict"),
        "",
        "## Context",
        "",
        "Step 15J showed that broad global-association relaxation reduced fragmentation only modestly while increasing false merges.",
        "Person is the dominant fragmentation class, so Step 15K tests conservative Person-only cleanup policies.",
        "",
        "## Runs",
        "",
        "| run | Track1 errors | Track1 rows | Person rows | non-Person delta | Person frag | purity | false merge |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in runs:
        lines.append(
            "| %s | %s | %s | %s | %s | %s | %s | %s |"
            % (
                row.get("run_name"),
                _fmt(row.get("track1_validation_errors")),
                _fmt(row.get("track1_rows")),
                _fmt(row.get("person_rows")),
                _fmt(row.get("vs_v2_non_person_rows_delta")),
                _fmt(row.get("person_fragmentation_approx")),
                _fmt(row.get("global_purity_mean")),
                _fmt(row.get("false_merge_rate")),
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- A cleanup run is safer when non-Person rows remain unchanged and Track1 validation has zero errors.",
            "- Pruning can reduce row inflation, but it may also remove valid short Person detections.",
            "- Selective merge is diagnostic-first unless explicitly enabled with `apply_merges: true`.",
            "",
            "## Next Step",
            "",
            "Step 15L should promote only a cleanup run with clean Track1 validation, no non-Person impact, and a useful Person row/fragmentation reduction.",
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

