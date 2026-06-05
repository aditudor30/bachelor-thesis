"""Markdown reporting for Person-aware association experiments."""

from pathlib import Path
from typing import Any, Dict, List


def write_person_association_report(summary: Dict[str, Any], output_path: Path) -> None:
    """Write a concise markdown report."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    recommendation = summary.get("best_person_association_recommendation", {})
    lines.append("# Person-aware Association Report")
    lines.append("")
    lines.append("## Verdict")
    lines.append("")
    lines.append("- verdict: `%s`" % recommendation.get("verdict"))
    lines.append("- best_run: `%s`" % recommendation.get("best_run"))
    lines.append("")
    lines.append("## Baselines")
    lines.extend(_baseline_lines("v2_current", summary.get("v2_current", {})))
    lines.extend(_baseline_lines("v2_export_compact", summary.get("v2_export_compact", {})))
    lines.extend(_baseline_lines("v1_baseline", summary.get("v1_baseline", {})))
    lines.append("")
    lines.append("## Runs")
    lines.append("")
    for run in summary.get("runs", []):
        lines.append("- `%s`: track1_rows=%s, person_frag=%s, purity=%s, false_merge=%s, non_person_delta=%s, merges=%s" % (
            run.get("run_name"),
            run.get("track1_rows"),
            run.get("person_fragmentation_approx"),
            run.get("global_purity_mean"),
            run.get("false_merge_rate"),
            run.get("vs_v2_non_person_rows_delta"),
            run.get("applied_merge_mapping_size"),
        ))
    lines.append("")
    lines.append("## Recommendation For Step 15M")
    lines.append("")
    lines.append("Use the selected run only if Track1 validation is clean, non-Person rows are unchanged, and the purity/false-merge trade-off stays within the configured limits. If the verdict is minor or not beneficial, the next useful upgrade is appearance/ReID-guided Person association rather than broader geometry relaxation.")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _baseline_lines(name: str, metrics: Dict[str, Any]) -> List[str]:
    if not metrics:
        return ["", "- `%s`: unavailable" % name]
    return [
        "",
        "- `%s`: track1_rows=%s, person_rows=%s, non_person_rows=%s, person_frag=%s, purity=%s, false_merge=%s"
        % (
            name,
            metrics.get("track1_rows"),
            metrics.get("person_rows"),
            metrics.get("non_person_rows"),
            metrics.get("person_fragmentation_approx"),
            metrics.get("global_purity_mean"),
            metrics.get("false_merge_rate"),
        ),
    ]

