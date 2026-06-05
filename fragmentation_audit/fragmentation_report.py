"""Markdown report generation for the fullcam fragmentation audit."""

from pathlib import Path
from typing import Any, Dict, List

from deep_oc_sort_3d.fragmentation_audit.fragmentation_io import write_json


def write_fragmentation_report(
    comparison: Dict[str, Any],
    root_cause: Dict[str, Any],
    stage_results: Dict[str, Dict[str, Any]],
    output_root: Path,
) -> None:
    """Write Markdown and JSON report outputs."""
    report_root = output_root / "report"
    report_root.mkdir(parents=True, exist_ok=True)
    write_json(
        {
            "comparison": comparison,
            "root_cause": root_cause,
            "stage_results": stage_results,
        },
        report_root / "BASELINE_V2_FULLCAM_FRAGMENTATION_AUDIT_SUMMARY.json",
    )
    text = build_report_markdown(comparison, root_cause, stage_results)
    (report_root / "BASELINE_V2_FULLCAM_FRAGMENTATION_AUDIT_REPORT.md").write_text(text, encoding="utf-8")
    write_optional_plots(comparison, stage_results, output_root / "plots_optional")


def build_report_markdown(
    comparison: Dict[str, Any],
    root_cause: Dict[str, Any],
    stage_results: Dict[str, Dict[str, Any]],
) -> str:
    """Build a Markdown report."""
    lines = []
    lines.append("# Baseline V2 Fullcam Fragmentation Audit")
    lines.append("")
    lines.append("## Executive Summary")
    lines.append("")
    lines.append("* Verdict: `%s`" % root_cause.get("verdict", "not_available"))
    lines.append("* This report is diagnostic only; it does not tune tracking parameters.")
    lines.append("* Metrics marked `None` or `not_available` could not be computed from available outputs.")
    lines.append("")
    lines.append("## Step 15H Recap")
    lines.append("")
    lines.append("* Baseline V2 fullcam is Track1-valid and pseudo3D-backed at high coverage.")
    lines.append("* The known issue is increased fragmentation, so this audit localizes where it appears.")
    lines.append("")
    lines.append("## High-Level Deltas")
    lines.append("")
    for key, value in sorted(comparison.get("high_level", {}).items()):
        lines.append("* `%s`: %s" % (key, value))
    lines.append("")
    lines.append("## Per-Stage Analysis")
    lines.append("")
    for stage in _stage_order(stage_results):
        lines.append("### %s" % stage)
        result = comparison.get("stages", {}).get(stage, {})
        for row in result.get("rows", []):
            lines.append(
                "* `%s`: V1=%s, V2=%s, delta=%s"
                % (row.get("metric"), row.get("baseline_v1"), row.get("baseline_v2_fullcam"), row.get("delta"))
            )
        lines.append("")
    lines.append("## Root-Cause Hypothesis")
    lines.append("")
    for reason in root_cause.get("reasons", []):
        lines.append("* %s" % reason)
    if not root_cause.get("reasons"):
        lines.append("* No dominant cause could be inferred from available metrics.")
    lines.append("")
    lines.append("## Concrete Tuning Plan For Step 15J")
    lines.append("")
    for rec in root_cause.get("tuning_recommendations", []):
        lines.append("* `%s`: %s Watch: %s" % (rec.get("area"), rec.get("action"), rec.get("metrics_to_watch")))
    lines.append("")
    lines.append("## Limitations")
    lines.append("")
    lines.append("* GT-based purity/fragmentation is available only where previous outputs preserved GT diagnostics.")
    lines.append("* Test split has no GT/depth, so test conclusions are structural rather than GT-evaluated.")
    lines.append("* Optional plots are skipped when matplotlib is unavailable.")
    lines.append("")
    return "\n".join(lines)


def write_optional_plots(comparison: Dict[str, Any], stage_results: Dict[str, Dict[str, Any]], output_root: Path) -> None:
    """Generate optional simple plots when matplotlib is installed."""
    try:
        import matplotlib.pyplot as plt
    except Exception:
        output_root.mkdir(parents=True, exist_ok=True)
        (output_root / "plots_skipped.txt").write_text("matplotlib unavailable\n", encoding="utf-8")
        return
    output_root.mkdir(parents=True, exist_ok=True)
    _bar_plot(
        plt,
        comparison.get("high_level", {}),
        "High-level fragmentation deltas",
        output_root / "fragmentation_high_level_deltas.png",
    )
    _bar_plot(
        plt,
        stage_results.get("global_association", {}).get("baseline_v2", {}).get("per_class", {}),
        "V2 global tracks by class",
        output_root / "fragmentation_by_class.png",
    )
    _bar_plot(
        plt,
        stage_results.get("global_association", {}).get("baseline_v2", {}).get("per_scene", {}),
        "V2 global tracks by scene",
        output_root / "fragmentation_by_scene.png",
    )


def _bar_plot(plt: Any, data: Dict[str, Any], title: str, path: Path) -> None:
    keys = []
    values = []
    for key, value in data.items():
        try:
            val = float(value)
        except (TypeError, ValueError):
            continue
        keys.append(str(key))
        values.append(val)
    if not values:
        return
    plt.figure(figsize=(12, 5))
    plt.bar(range(len(values)), values)
    plt.xticks(range(len(values)), keys, rotation=45, ha="right")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(str(path))
    plt.close()


def _stage_order(stage_results: Dict[str, Dict[str, Any]]) -> List[str]:
    preferred = [
        "observations",
        "local_tracking",
        "tracklets",
        "candidates",
        "motion_filtering",
        "global_association",
        "final_export",
    ]
    return [stage for stage in preferred if stage in stage_results] + sorted([stage for stage in stage_results if stage not in preferred])

