"""Honest Markdown report for Step 22D."""

from pathlib import Path
from typing import Any, Dict, List

from deep_oc_sort_3d.v5_geometry_calibration.geometry_calibration_config import output_root
from deep_oc_sort_3d.v5_geometry_calibration.geometry_calibration_io import read_csv, read_json


def write_v5_report(config: Dict[str, Any]) -> Path:
    """Write the requested calibration, validation and upload report."""
    root = output_root(config)
    comparison = read_json(root / "comparison" / "v5_geometry_calibration_summary.json")
    availability = read_json(root / "audit" / "input_availability_audit.json")
    input_summary = read_json(root / "audit" / "v4_input_summary.json")
    dataset = read_json(root / "calibration_dataset" / "calibration_matches_summary.json")
    selected = read_json(root / "learned_corrections" / "selected_corrections.json")
    rejected = read_json(root / "learned_corrections" / "rejected_corrections.json")
    readiness = read_json(root / "frozen_candidate" / "comparison" / "upload_readiness.json")
    package = readiness.get("v5_geometry_calibrated_official", {})
    lines: List[str] = [
        "# V5 Geometry Calibration Official 023-027 Report", "",
        "## Executive Summary", "",
        "V5 estimates conservative class-level geometry corrections from train/val pseudo3D-to-GT matches and applies only selected corrections to the validated V4 input or its V3.1 fallback.",
        "GT and depth are never read for test scenes 023-027. Test coverage, IDs, classes and frame keys remain immutable.", "",
        "## Decision", "",
        "- Selected variant: `%s`" % (comparison.get("selected_variant") or "none"),
        "- Verdict: `%s`" % comparison.get("verdict", "unknown"),
        "- Upload ready: `%s`" % package.get("ready", False),
        "- Rows / unique tracks: `%s` / `%s`" % (package.get("rows"), package.get("unique_tracks")),
        "- Track1: `%s`" % package.get("track1_path", "not_available"),
        "- ZIP: `%s`" % package.get("zip_path", "not_available"), "",
        "## Input And Sources", "",
        "- Input variant: `%s`" % availability.get("input_variant"),
        "- Input Track1: `%s`" % input_summary.get("track1_path"),
        "- Input validation: `%s`, errors `%s`" % (input_summary.get("validation_status"), input_summary.get("validation_errors")),
        "- Available calibration scenes: `%s`" % availability.get("available_calibration_scenes"),
        "- Missing calibration scenes: `%s`" % availability.get("missing_calibration_scenes"),
        "- Observation source files: `%s`" % dataset.get("source_files"), "",
        "## Calibration Dataset", "",
        "- Predictions scanned: `%s`" % dataset.get("num_predictions"),
        "- GT-visible count proxy: `%s`" % dataset.get("num_gt"),
        "- Conservative matches: `%s`" % dataset.get("num_matches"),
        "- Match rate: `%s`" % dataset.get("match_rate"),
        "- Ambiguous matches rejected: `%s`" % dataset.get("ambiguous_matches_rejected"),
        "- Samples per class: `%s`" % dataset.get("samples_per_class"),
        "- Samples per phase: `%s`" % dataset.get("samples_per_phase"), "",
        "## Corrections", "",
        "- Fit source: `%s`" % selected.get("fit_source"),
        "- Selected dimension classes: `%s`" % _selected_classes(selected.get("dimension", {})),
        "- Selected center classes: `%s`" % _selected_classes(selected.get("center", {})),
        "- Selected yaw classes: `%s`" % _selected_classes(selected.get("yaw", {})),
        "- Depth calibration: `not_applied_due_to_missing_camera_mapping`", 
        "- Rejected correction records: `%s`" % len(rejected.get("corrections", [])), "",
        "### Rejection Reasons", "",
        "%s" % _rejection_reason_text(rejected), "",
        "## Train / Holdout / Official Val Before-After", "",
        "The table below reports the selected test variant when available; otherwise it reports the balanced diagnostic correction set.", "",
        "| Phase | Metric | Before | After | Delta | Samples |",
        "|---|---|---:|---:|---:|---:|",
    ]
    report_variant = str(comparison.get("selected_variant") or "v5_geometry_calibrated_balanced")
    for phase, filename in [
        ("fit_train", "train_eval_before_after.csv"),
        ("internal_holdout", "holdout_eval_before_after.csv"),
        ("official_val", "official_val_eval_before_after.csv"),
    ]:
        for row in _diagnostic_rows(root / "validation_diagnostics" / filename, report_variant):
            lines.append("| %s | %s | %s | %s | %s | %s |" % (
                phase, row.get("metric"), _fmt(row.get("before")), _fmt(row.get("after")),
                _fmt(row.get("delta")), row.get("samples"),
            ))
    lines.extend([
        "",
        "## Variant Validation", "",
        "| Variant | Valid | Risk | Useful | Rows | Tracks | Position p95 | Dimension ratio p95 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for row in comparison.get("variants", []):
        lines.append("| %s | %s | %s | %s | %s | %s | %s | %s |" % (
            row.get("variant"), row.get("hard_valid"), row.get("quality_risk"), row.get("has_selected_correction"),
            row.get("rows"), row.get("unique_tracks"), _fmt(row.get("p95_position_change_m")),
            _fmt(row.get("p95_dimension_change_ratio")),
        ))
    lines.extend([
        "", "## Interpretation Limits", "",
        "Train/val improvements are diagnostic and depend on the available pseudo3D observation coverage. Missing fit scenes reduce confidence and are reported rather than silently replaced.",
        "Camera-specific and ray/depth calibration is disabled because official Track1 rows do not provide a reliable camera mapping.",
        "Official test GT is unavailable, so only the challenge server can confirm whether V5 improves the official 3D metric.", "",
        "## Recommendation", "",
        "Upload V5 only when it is marked ready and the submission budget allows another geometry-calibrated candidate. Keep V4/V3.1 as the safer reference when gains are small or source coverage is limited.",
    ])
    path = root / "comparison" / "V5_GEOMETRY_CALIBRATION_OFFICIAL_023_027_REPORT.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _selected_classes(values: Dict[str, Any]) -> List[int]:
    return sorted([int(key) for key, item in values.items() if item.get("selected")])


def _diagnostic_rows(path: Path, variant: str) -> List[Dict[str, Any]]:
    metrics = {
        "center_error_mean", "center_error_median", "dimension_error_mean", "dimension_error_median",
        "yaw_error_mean", "yaw_error_median", "3d_iou_proxy_mean", "3d_iou_proxy_median",
    }
    return [row for row in read_csv(path) if row.get("variant") == variant and row.get("metric") in metrics]


def _rejection_reason_text(rejected: Dict[str, Any]) -> str:
    counts: Dict[str, int] = {}
    for item in rejected.get("corrections", []):
        reason = str(item.get("reject_reason", "unknown"))
        counts[reason] = counts.get(reason, 0) + 1
    if not counts:
        return "No rejected corrections were recorded."
    return ", ".join("`%s`: %d" % (key, counts[key]) for key in sorted(counts.keys()))


def _fmt(value: Any) -> str:
    try:
        return "%.6f" % float(value)
    except (TypeError, ValueError):
        return "not_available"
