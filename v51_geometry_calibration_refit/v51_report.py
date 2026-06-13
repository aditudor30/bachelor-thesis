"""Honest Step 22E report covering sources, split metrics and upload readiness."""

from pathlib import Path
from typing import Any, Dict, List

from deep_oc_sort_3d.v51_geometry_calibration_refit.v51_config import output_root
from deep_oc_sort_3d.v51_geometry_calibration_refit.v51_io import read_csv, read_json


def write_v51_report(config: Dict[str, Any]) -> Path:
    root = output_root(config)
    availability = read_json(root / "audit" / "input_availability_audit.json")
    generation = read_json(root / "audit" / "generated_train_sources_summary.json")
    dataset = read_json(root / "calibration_dataset" / "match_rate_summary.json")
    selected = read_json(root / "learned_corrections" / "selected_corrections.json")
    rejected = read_json(root / "learned_corrections" / "rejected_corrections.json")
    comparison = read_json(root / "comparison" / "v51_geometry_calibration_summary.json")
    readiness = read_json(root / "frozen_candidate" / "comparison" / "upload_readiness.json")
    package = readiness.get("v51_geometry_calibrated_official", {})
    lines: List[str] = [
        "# V5.1 Geometry Calibration Refit Official 023-027 Report", "",
        "## Executive Summary", "",
        "V5.1 exists to replace V5's internal-holdout fitting fallback with a strict fit on Warehouse_000-013, tuning on Warehouse_014-019, and final diagnostics on Warehouse_020-022.",
        "GT is used only for train/val matching and diagnostics. Test scenes 023-027 are never read from GT or depth, and Track1 identities and coverage are immutable.", "",
        "## Decision", "",
        "- Selected variant: `%s`" % (comparison.get("selected_variant") or "none"),
        "- Verdict: `%s`" % comparison.get("verdict", "unknown"),
        "- Fit source: `%s`" % selected.get("fit_source", "not_available"),
        "- Upload ready: `%s`" % package.get("ready", False),
        "- Rows / unique tracks: `%s` / `%s`" % (package.get("rows"), package.get("unique_tracks")),
        "- Track1: `%s`" % package.get("track1_path", "not_available"),
        "- ZIP: `%s`" % package.get("zip_path", "not_available"), "",
        "## Why V5.1 Was Needed", "",
        "V5 reported `fit_source = internal_holdout_fallback`. V5.1 forbids that fallback; when Warehouse_000-013 cannot be used, no V5.1 candidate is selected and V5 remains the calibrated reference.", "",
        "## Fit-Train Sources", "",
        "- Existing fit_train complete before generation: `%s`" % (not bool(generation.get("missing_before_generation"))),
        "- Generated missing sources: `%s`" % generation.get("generated", False),
        "- Generation status: `%s`" % generation.get("status", "not_available"),
        "- Missing after generation: `%s`" % generation.get("missing_after_generation", []),
        "- Detector checkpoint: `%s`" % generation.get("detector_checkpoint", "not_used"),
        "- GT used for generated predictions: `%s`" % generation.get("gt_used_for_predictions", False),
        "- Depth used for generated predictions: `%s`" % generation.get("depth_used_for_predictions", False),
        "- Available calibration scenes: `%s`" % availability.get("available_calibration_scenes"), "",
        "## Calibration Dataset", "",
        "| Split | Predictions | GT-visible proxy | Matches | Match rate |",
        "|---|---:|---:|---:|---:|",
    ]
    for phase in ["fit_train", "internal_holdout", "official_val"]:
        item = dataset.get(phase, {})
        lines.append("| %s | %s | %s | %s | %s |" % (
            phase, item.get("num_predictions"), item.get("num_gt"), item.get("num_matches"), _fmt(item.get("match_rate")),
        ))
    lines.extend([
        "", "- Ambiguous matches rejected: `%s`" % dataset.get("ambiguous_matches_rejected"),
        "- Samples per class: `%s`" % dataset.get("samples_per_class"),
        "- Samples per scene: `%s`" % dataset.get("samples_per_scene"), "",
        "## Learned And Rejected Corrections", "",
        "- Selected dimension classes: `%s`" % _selected_classes(selected.get("dimension", {})),
        "- Selected center classes: `%s`" % _selected_classes(selected.get("center", {})),
        "- Selected yaw classes: `%s`" % _selected_classes(selected.get("yaw", {})),
        "- Depth: `camera_specific_calibration_not_applied`",
        "- Rejected corrections: `%s`" % len(rejected.get("corrections", [])),
        "- Rejection reasons: %s" % _rejection_reasons(rejected), "",
        "## Before / After", "",
        "| Split | Metric | Before | After | Delta | Samples |",
        "|---|---|---:|---:|---:|---:|",
    ])
    report_variant = str(comparison.get("selected_variant") or "v51_geometry_refit_balanced")
    for phase, filename in [
        ("fit_train", "fit_train_before_after.csv"),
        ("internal_holdout", "internal_holdout_before_after.csv"),
        ("official_val", "official_val_before_after.csv"),
    ]:
        for row in _diagnostic_rows(root / "validation_diagnostics" / filename, report_variant):
            lines.append("| %s | %s | %s | %s | %s | %s |" % (
                phase, row.get("metric"), _fmt(row.get("before")), _fmt(row.get("after")),
                _fmt(row.get("delta")), row.get("samples"),
            ))
    lines.extend([
        "", "## V5 Versus V5.1", "",
        "V5.1 is preferred only when its fit source is `fit_train`, at least one official-val geometry metric improves, Track1 validation has zero errors, and all test-change safety gates pass.",
        "When these conditions fail, the verdict intentionally keeps V5 as the safer calibrated candidate.", "",
        "## Recommendation", "",
        _recommendation(comparison, package),
    ])
    path = root / "comparison" / "V51_GEOMETRY_CALIBRATION_REFIT_OFFICIAL_023_027_REPORT.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _diagnostic_rows(path: Path, variant: str) -> List[Dict[str, Any]]:
    metrics = {
        "center_error_mean", "center_error_median", "center_error_p95",
        "dimension_error_mean", "dimension_error_median", "dimension_error_p90",
        "yaw_error_mean", "yaw_error_median", "yaw_error_p90",
        "3d_iou_proxy_mean", "3d_iou_proxy_median",
    }
    return [row for row in read_csv(path) if row.get("variant") == variant and row.get("metric") in metrics]


def _selected_classes(values: Dict[str, Any]) -> List[int]:
    return sorted(int(key) for key, item in values.items() if item.get("selected"))


def _rejection_reasons(rejected: Dict[str, Any]) -> str:
    counts: Dict[str, int] = {}
    for item in rejected.get("corrections", []):
        reason = str(item.get("reject_reason", "unknown"))
        counts[reason] = counts.get(reason, 0) + 1
    return ", ".join("`%s`: %d" % (key, counts[key]) for key in sorted(counts)) or "none"


def _recommendation(comparison: Dict[str, Any], package: Dict[str, Any]) -> str:
    if package.get("ready"):
        return "V5.1 is upload-ready. Use its exact frozen ZIP only if the submission budget permits another robust geometry candidate."
    if comparison.get("verdict") == "v51_invalid_missing_fit_train_sources":
        return "Warehouse_000-013 could not support a true refit. Do not upload V5.1; V5 remains the safer calibrated candidate."
    return "V5.1 is not upload-ready under the configured gates. Keep V5 as the safer calibrated candidate."


def _fmt(value: Any) -> str:
    try:
        return "%.6f" % float(value)
    except (TypeError, ValueError):
        return "not_available"
