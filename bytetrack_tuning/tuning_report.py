"""Comparison artifacts and Markdown report for ByteTrack coverage tuning."""

from pathlib import Path
from typing import Any, Dict, List

from deep_oc_sort_3d.bytetrack_tuning.coverage_drop_analyzer import compute_coverage_drop_rows
from deep_oc_sort_3d.bytetrack_tuning.stage_retention_analyzer import (
    build_dimension_retention_rows,
    build_stage_retention_rows,
)
from deep_oc_sort_3d.bytetrack_tuning.tuning_config import tuning_output_root
from deep_oc_sort_3d.bytetrack_tuning.tuning_figures import write_tuning_figures
from deep_oc_sort_3d.bytetrack_tuning.tuning_io import write_csv, write_json
from deep_oc_sort_3d.bytetrack_tuning.tuning_metrics import collect_all_tuning_metrics
from deep_oc_sort_3d.bytetrack_tuning.tuning_selector import select_tuned_variant


def compare_tuning_runs(config: Dict[str, Any], progress: bool = True) -> Dict[str, Any]:
    """Collect metrics, select the best valid variant and write all artifacts."""
    root = tuning_output_root(config)
    metrics = collect_all_tuning_metrics(config, include_baselines=True, progress=progress)
    retention_rows = build_stage_retention_rows(metrics)
    per_scene = build_dimension_retention_rows(metrics, "scene")
    per_class = build_dimension_retention_rows(metrics, "class")
    per_camera = build_dimension_retention_rows(metrics, "camera")
    person_nonperson = build_dimension_retention_rows(metrics, "person_vs_nonperson")
    coverage_rows = compute_coverage_drop_rows(metrics.get("variants", {}))
    selection = select_tuned_variant(metrics, config)
    summary_rows = selection.get("variant_selection_rows", [])

    diagnostics = root / "diagnostics"
    comparison = root / "comparison"
    write_csv(diagnostics / "stage_retention_summary.csv", retention_rows)
    write_json(diagnostics / "stage_retention_summary.json", {"rows": retention_rows})
    write_csv(diagnostics / "per_scene_retention.csv", per_scene)
    write_csv(diagnostics / "per_class_retention.csv", per_class)
    write_csv(diagnostics / "per_camera_retention.csv", per_camera)
    write_csv(diagnostics / "person_vs_nonperson_retention.csv", person_nonperson)
    write_csv(diagnostics / "coverage_drop_analysis.csv", coverage_rows)
    warnings = _warnings(metrics, config)
    write_json(diagnostics / "warnings.json", {"warnings": warnings})

    write_csv(comparison / "sweep_summary.csv", summary_rows)
    write_json(comparison / "sweep_summary.json", {"rows": summary_rows})
    write_csv(comparison / "metric_deltas_vs_v2_current.csv", _metric_delta_rows(metrics, "baseline_v2_current"))
    write_csv(comparison / "metric_deltas_vs_21b_bytetrack.csv", _metric_delta_rows(metrics, "baseline_21b_bytetrack"))
    write_csv(comparison / "metric_deltas_vs_v1.csv", _metric_delta_rows(metrics, "baseline_v1_geometry_only"))
    selected_config = _selected_config(selection, config)
    write_json(comparison / "selected_bytetrack_config.json", selected_config)
    write_json(comparison / "verdict.json", selection.get("verdict", {}))
    result = {
        "metrics": metrics,
        "selection": selection,
        "warnings": warnings,
        "stage_retention_rows": retention_rows,
        "coverage_drop_rows": coverage_rows,
    }
    write_markdown_report(result, comparison / "BYTETRACK_COVERAGE_TUNING_REPORT.md")
    if bool(config.get("figures", {}).get("enabled", True)):
        write_tuning_figures(summary_rows, per_class, root / "figures")
    return result


def write_markdown_report(result: Dict[str, Any], path: Path) -> None:
    """Write an honest, compact Step 21C report."""
    selection = result.get("selection", {})
    selected = selection.get("selected_metrics") or {}
    verdict = selection.get("verdict", {})
    rows = selection.get("variant_selection_rows", [])
    lines = [
        "# ByteTrack Coverage Tuning Report",
        "",
        "## Context",
        "",
        "Step 21B improved continuity and purity but retained only 26.3% of Track1 rows and 11.4% of multi-camera tracks. Step 21C therefore treats coverage as a hard gate, not a secondary metric.",
        "",
        "## Sweep",
        "",
        "| Variant | Hard gates | Local retention | GT retention | Track1 retention | Multi-camera retention | Score |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| %s | %s | %s | %s | %s | %s | %s |"
            % (
                row.get("variant"),
                "pass" if row.get("hard_criteria_met") else "fail",
                _fmt(row.get("local_records_retention")),
                _fmt(row.get("gt_matched_retention")),
                _fmt(row.get("track1_rows_retention")),
                _fmt(row.get("multi_camera_tracks_retention")),
                _fmt(row.get("selection_score")),
            )
        )
    lines.extend(
        [
            "",
            "## Selected configuration",
            "",
            "- Variant: `%s`" % selection.get("selected_variant"),
            "- Track1 validation errors: `%s`" % selected.get("track1_validation_errors"),
            "- Local records retention: `%s`" % _fmt(selected.get("local_records_retention")),
            "- GT matched retention: `%s`" % _fmt(selected.get("gt_matched_retention")),
            "- Track1 rows retention: `%s`" % _fmt(selected.get("track1_rows_retention")),
            "- Multi-camera retention: `%s`" % _fmt(selected.get("multi_camera_tracks_retention")),
            "- Purity delta: `%s`" % _fmt(selected.get("purity_delta")),
            "- False-merge delta: `%s`" % _fmt(selected.get("false_merge_rate_delta")),
            "",
            "## Verdict",
            "",
            "`%s`" % verdict.get("label"),
            "",
            "Reasons: %s" % ", ".join(verdict.get("reasons", [])),
            "",
            "## Interpretation",
            "",
            "- A variant is selected only after passing coverage, validation, purity and false-merge gates.",
            "- Phase-A-only variants cannot become the final candidate because they have no Track1/global metrics.",
            "- Person-specific thresholds are recorded but intentionally not applied in this implementation; all runs remain class-safe and use one common threshold set.",
            "",
            "## Next step",
            "",
            _next_step(verdict.get("label")),
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _metric_delta_rows(metrics: Dict[str, Any], baseline_name: str) -> List[Dict[str, Any]]:
    baseline = metrics.get("baselines", {}).get(baseline_name, {})
    paths = {
        "local_records": ("local_tracking", "num_records"),
        "local_tracks": ("local_tracking", "num_tracks"),
        "median_track_length": ("local_tracking", "median_track_length"),
        "short_track_ratio_le3": ("local_tracking", "short_track_ratio_le3"),
        "local_fragmentation": ("local_tracking", "approx_fragmentation"),
        "local_id_switches": ("local_tracking", "approx_id_switches"),
        "global_tracks": ("global_association", "global_tracks"),
        "multi_camera_tracks": ("global_association", "multi_camera_tracks"),
        "global_purity_mean": ("global_association", "global_purity_mean"),
        "false_merge_rate": ("global_association", "false_merge_rate"),
        "global_fragmentation": ("global_association", "fragmentation_approx"),
        "track1_rows": ("track1", "rows"),
    }
    rows = []
    for variant_name, variant in sorted(metrics.get("variants", {}).items()):
        for metric, metric_path in paths.items():
            left = _nested(baseline, metric_path)
            right = _nested(variant, metric_path)
            rows.append(
                {
                    "variant": variant_name,
                    "baseline": baseline_name,
                    "metric": metric,
                    "baseline_value": left,
                    "variant_value": right,
                    "delta": _delta(left, right),
                }
            )
    return rows


def _selected_config(selection: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    name = selection.get("selected_variant")
    return {
        "selected_variant": name,
        "variant_config": config.get("variants", {}).get(name, {}) if name is not None else None,
        "selected_metrics": selection.get("selected_metrics"),
        "person_specific_applied": False,
    }


def _warnings(metrics: Dict[str, Any], config: Dict[str, Any]) -> List[str]:
    output = []
    if config.get("variants", {}).get("bt_coverage_person_friendly", {}).get("person_specific"):
        output.append("person_specific thresholds were requested but not applied; the variant uses its common thresholds")
    for name, values in metrics.get("variants", {}).items():
        if values.get("status") in ("missing", "error"):
            output.append("%s has status %s" % (name, values.get("status")))
    return output


def _nested(data: Dict[str, Any], path: Any) -> Any:
    current = data
    for key in path:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _delta(left: Any, right: Any) -> Any:
    try:
        return float(right) - float(left)
    except (TypeError, ValueError):
        return None


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    try:
        return "%.4f" % float(value)
    except (TypeError, ValueError):
        return str(value)


def _next_step(label: Any) -> str:
    if label == "bytetrack_tuned_ready_for_full_submission_candidate":
        return "Proceed to Step 21D: freeze the selected config, rerun/final-check the full candidate, and package it beside V2 current."
    if label == "bytetrack_tuned_valid_coverage_recovered_needs_global_tuning":
        return "Proceed to a narrow Step 21D global-association tuning around the selected local tracker."
    return "Do not replace V2 current. Inspect coverage_drop_analysis.csv and revise only the stage responsible for the largest loss."
