"""Comparison orchestration and honest final report for Step 21E."""

from pathlib import Path
from typing import Any, Dict, List

from deep_oc_sort_3d.bytetrack_motion_filtering.motion_filter_config import output_root
from deep_oc_sort_3d.bytetrack_motion_filtering.motion_filter_figures import write_motion_filter_figures
from deep_oc_sort_3d.bytetrack_motion_filtering.motion_filter_io import read_csv, read_json
from deep_oc_sort_3d.bytetrack_motion_filtering.motion_filter_metrics import collect_motion_filter_comparison
from deep_oc_sort_3d.bytetrack_motion_filtering.motion_filter_selector import select_motion_filter_variant


def compare_and_report(config: Dict[str, Any], progress: bool = True) -> Dict[str, Any]:
    """Collect metrics, select a variant and write the report and figures."""
    if progress:
        print("Step 21E comparison: collecting pipeline metrics")
    comparison = collect_motion_filter_comparison(config)
    if progress:
        print("Step 21E comparison: applying safety gates")
    selection = select_motion_filter_variant(config, comparison)
    figures = write_motion_filter_figures(config, comparison)
    report_path = output_root(config) / "comparison" / "BYTETRACK_GAP_AWARE_MOTION_FILTER_REPORT.md"
    write_motion_filter_report(config, comparison, selection, figures, report_path)
    return {"comparison": comparison, "selection": selection, "figures": figures, "report": str(report_path)}


def write_motion_filter_report(
    config: Dict[str, Any],
    comparison: Dict[str, Any],
    selection: Dict[str, Any],
    figures: List[str],
    path: Path,
) -> None:
    """Write the requested Step 21E Markdown report."""
    rows = comparison.get("rows", [])
    priors = read_json(output_root(config) / "velocity_priors" / "class_velocity_priors.json").get("classes", {})
    rejected_gap = read_csv(output_root(config) / "diagnostics" / "rejected_by_gap_bucket.csv")
    rejected_class = read_csv(output_root(config) / "diagnostics" / "rejected_by_class.csv")
    rejected_scene = read_csv(output_root(config) / "diagnostics" / "rejected_by_scene.csv")
    verdict = str(selection.get("verdict"))
    selected = selection.get("selected_variant")
    lines = [
        "# ByteTrack Gap-Aware Motion Filter Report",
        "",
        "## Executive summary",
        "",
        "Verdict: `%s`" % verdict,
        "",
        "Selected variant: `%s`" % (selected or "none"),
        "",
        "Recommended Step 21F: %s" % selection.get("recommended_step_21f"),
        "",
        "Reasons: %s" % ", ".join(selection.get("reasons", [])),
        "",
        "## Context 21B, 21C and 21D",
        "",
        "ByteTrack improved local continuity but retained too little final coverage. Step 21D isolated motion filtering as the main downstream candidate loss, including substantial rejection at small and medium temporal gaps.",
        "",
        "## Velocity priors",
        "",
        "Priors are world displacement per frame. GT is used only on configured train/validation scenes; test uses the frozen priors or explicit fallbacks.",
        "",
    ]
    for name, prior in sorted(priors.items()):
        lines.append(
            "- %s: v_max=%s, margin=%s, cap=%s, samples=%s, source=%s, confidence=%s"
            % (
                name,
                _fmt(prior.get("recommended_v_max")),
                _fmt(prior.get("recommended_margin")),
                _fmt(prior.get("absolute_cap")),
                prior.get("num_samples"),
                prior.get("source"),
                prior.get("confidence"),
            )
        )
    lines.extend(["", "## Variants and final metrics", ""])
    for row in rows:
        lines.append(
            "- `%s`: motion retention=%s, Track1 rows=%s, Track1 errors=%s, multi-camera=%s, purity=%s, false merges=%s, fragmentation=%s"
            % (
                row.get("variant_name"),
                _fmt(row.get("motion_clean_retention")),
                row.get("track1_rows"),
                row.get("track1_validation_errors"),
                row.get("multi_camera_tracks"),
                _fmt(row.get("global_purity_mean")),
                _fmt(row.get("false_merge_rate")),
                row.get("fragmentation_approx"),
            )
        )
    lines.extend(["", "## Rejection by gap bucket", ""])
    for row in rejected_gap:
        lines.append("- %s / %s: rejected rate=%s" % (row.get("variant_name"), row.get("gap_bucket"), _fmt(row.get("rate"))))
    lines.extend(["", "## Rejection by class", ""])
    for row in rejected_class:
        lines.append("- %s / %s: rejected rate=%s" % (row.get("variant_name"), row.get("class_name"), _fmt(row.get("rate"))))
    lines.extend(["", "## Rejection by scene", ""])
    for row in rejected_scene:
        lines.append("- %s / %s: rejected rate=%s" % (row.get("variant_name"), row.get("scene_name"), _fmt(row.get("rate"))))
    lines.extend(["", "## Baseline comparison", ""])
    for baseline_name, baseline in sorted(comparison.get("baselines", {}).items()):
        lines.append(
            "- `%s`: motion retention=%s, Track1 rows=%s, multi-camera=%s, purity=%s, false merges=%s, fragmentation=%s"
            % (
                baseline_name,
                _fmt(baseline.get("motion", {}).get("motion_clean_retention")),
                baseline.get("track1", {}).get("rows"),
                baseline.get("global", {}).get("multi_camera_tracks"),
                _fmt(baseline.get("global", {}).get("global_purity_mean")),
                _fmt(baseline.get("global", {}).get("false_merge_rate")),
                baseline.get("global", {}).get("fragmentation_approx"),
            )
        )
    lines.extend(
        [
            "",
            "## Pseudo-3D and bbox jump tolerance",
            "",
            "The bbox-tolerant variants only retain a suspicious jump when it remains below the class absolute cap and the total tolerated jump budget is not exceeded. These diagnostics do not use GT or depth on test.",
            "",
            "## Selection",
            "",
            "The selector requires zero Track1 validation errors, improved motion-clean retention, more Track1 rows than ByteTrack 21C, controlled purity loss and controlled false-merge increase.",
            "",
            "## Honest interpretation",
            "",
            _interpretation(verdict),
            "",
            "## Figures",
            "",
        ]
    )
    for figure in figures:
        lines.append("- `%s`" % figure)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def summarize_existing_output(root: Path) -> Dict[str, Any]:
    """Load compact Step 21E outputs for the summary CLI."""
    selected = read_json(root / "comparison" / "selected_motion_filter_variant.json")
    verdict = read_json(root / "comparison" / "verdict.json")
    rows = read_csv(root / "comparison" / "motion_filter_sweep_summary.csv")
    return {"selected": selected, "verdict": verdict, "rows": rows}


def _interpretation(verdict: str) -> str:
    if verdict == "gap_aware_motion_filter_ready_for_v3_candidate":
        return "The gap-aware filter recovers downstream coverage without violating the configured safety gates. It is suitable for a V3 candidate rerun, not yet an automatic replacement for the submission baseline."
    if verdict == "gap_aware_motion_filter_valid_improves_coverage_needs_global_tuning":
        return "Motion filtering improved, but the gain did not fully propagate through global association and Track1 export. The next bottleneck is downstream association tuning."
    if verdict == "gap_aware_motion_filter_valid_but_false_merges_too_high":
        return "Coverage was recovered at an unacceptable identity-merging cost. The relaxed filter must not be selected without tighter caps or stronger global safeguards."
    if verdict == "gap_aware_motion_filter_valid_but_small_gain":
        return "The change is valid but too small to justify replacing ByteTrack 21C by itself."
    if verdict == "gap_aware_motion_filter_invalid_fix_required":
        return "At least one required output or validation check failed. Fix the isolated 21E run before interpreting quality metrics."
    return "No variant delivered a clear safe gain over ByteTrack 21C. The audit remains useful, but Step 21F is not yet justified."


def _fmt(value: Any) -> str:
    if value in (None, ""):
        return "n/a"
    try:
        return "%.4f" % float(value)
    except (TypeError, ValueError):
        return str(value)
