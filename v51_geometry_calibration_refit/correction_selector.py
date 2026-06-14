"""Fit on Warehouse_000-013 and select corrections on holdout plus official val."""

import csv
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

from deep_oc_sort_3d.v51_geometry_calibration_refit.center_bias_refit import fit_center_biases
from deep_oc_sort_3d.v51_geometry_calibration_refit.depth_scale_refit import fit_depth_scales
from deep_oc_sort_3d.v51_geometry_calibration_refit.dimension_refit import fit_dimension_scales
from deep_oc_sort_3d.v51_geometry_calibration_refit.v51_config import VARIANT_NAMES, input_track1_path, output_root, variant_root
from deep_oc_sort_3d.v51_geometry_calibration_refit.v51_io import read_geometry_rows, read_json, write_csv, write_json
from deep_oc_sort_3d.v51_geometry_calibration_refit.v51_metrics import before_after_rows, evaluate_corrections, summarize_match_rows, summarize_test_changes
from deep_oc_sort_3d.v51_geometry_calibration_refit.yaw_bias_refit import fit_yaw_biases


COMPONENT_METRICS = {
    "dimension": "dimension_error_mean",
    "center": "center_error_mean",
    "yaw": "yaw_error_mean",
}


def fit_and_select_corrections(config: Dict[str, Any]) -> Dict[str, Any]:
    """Never fit on holdout: missing fit_train makes V5.1 ineligible."""
    root = output_root(config)
    dataset_root = root / "calibration_dataset"
    phase_files = {
        "fit_train": dataset_root / "fit_train_matches.csv",
        "internal_holdout": dataset_root / "internal_holdout_matches.csv",
        "official_val": dataset_root / "official_val_matches.csv",
    }
    phases: Dict[str, List[Dict[str, Any]]] = {}
    for phase, path in phase_files.items():
        print("V5.1 correction fitting: loading phase=%s path=%s" % (phase, path))
        phases[phase] = _read_numeric_rows(path)
        print("V5.1 correction fitting: loaded phase=%s rows=%d" % (phase, len(phases[phase])))
    fit_rows = phases.get("fit_train", [])
    fit_source = "fit_train" if fit_rows else "missing_fit_train"
    warnings: List[str] = []
    if not fit_rows:
        warnings.append("Warehouse_000-013 calibration matches are unavailable; internal_holdout fallback is forbidden")
    world_fit = _coordinate_safe_rows(fit_rows, config)
    learned = {
        "dimension": fit_dimension_scales(fit_rows, config) if fit_rows else {},
        "center": fit_center_biases(world_fit, config) if world_fit else {},
        "depth": fit_depth_scales(fit_rows, config) if fit_rows else {},
        "yaw": fit_yaw_biases(world_fit, config) if world_fit else {},
    }
    holdout_rows = phases.get("internal_holdout", [])
    val_rows = phases.get("official_val", [])
    selected_records: List[Dict[str, Any]] = []
    rejected_records: List[Dict[str, Any]] = []
    for component in ["dimension", "center", "yaw"]:
        metric = COMPONENT_METRICS[component]
        for class_id, item in learned[component].items():
            holdout = _class_rows(holdout_rows, class_id)
            official_val = _class_rows(val_rows, class_id)
            if component in ("center", "yaw"):
                holdout = _coordinate_safe_rows(holdout, config)
                official_val = _coordinate_safe_rows(official_val, config)
            candidate = _single_component(component, class_id, item)
            holdout_eval = _evaluation(holdout, candidate, metric)
            val_eval = _evaluation(official_val, candidate, metric)
            enough = bool(item.get("eligible_by_sample_count"))
            stable = bool(item.get("eligible_by_variability", True))
            holdout_improved = _improves(holdout_eval)
            val_improved = _improves(val_eval)
            accepted = fit_source == "fit_train" and enough and stable and holdout_improved and val_improved
            item["selected"] = accepted
            item["fit_source"] = fit_source
            item["internal_holdout_evaluation"] = holdout_eval
            item["official_val_evaluation"] = val_eval
            record = {"component": component, "official_class_id": int(class_id), "accepted": accepted, **item}
            if accepted:
                selected_records.append(record)
            else:
                record["reject_reason"] = _reject_reason(fit_source, enough, stable, holdout, official_val, holdout_improved, val_improved)
                rejected_records.append(record)
    for class_id, item in learned["depth"].items():
        item["selected"] = False
        item["fit_source"] = fit_source
        rejected_records.append({
            "component": "depth", "official_class_id": int(class_id), "accepted": False,
            "reject_reason": "camera_specific_calibration_not_applied", **item,
        })
    selected = {
        "fit_source": fit_source, "fallback_used": False, "warnings": warnings,
        "dimension": learned["dimension"], "center": learned["center"],
        "depth": learned["depth"], "yaw": learned["yaw"],
    }
    _write_corrections(root, learned, selected, rejected_records)
    diagnostics = _write_diagnostics(root, phases, selected)
    verdict = {
        "status": "ok" if fit_source == "fit_train" else "invalid_missing_fit_train_sources",
        "fit_source": fit_source, "selected_count": len(selected_records),
        "selected_components": sorted(set(item["component"] for item in selected_records)),
        "rejected_count": len(rejected_records), "warnings": warnings, "diagnostics": diagnostics,
    }
    write_json(root / "validation_diagnostics" / "calibration_verdict.json", verdict)
    return verdict


def correction_sets(selected: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    empty = {"dimension": {}, "center": {}, "depth": {}, "yaw": {}}
    return {
        "v51_dimension_scale_refit": dict(empty, dimension=selected.get("dimension", {})),
        "v51_center_bias_refit": dict(empty, center=selected.get("center", {})),
        "v51_depth_scale_refit": dict(empty, depth=selected.get("depth", {})),
        "v51_yaw_bias_refit": dict(empty, yaw=selected.get("yaw", {})),
        "v51_geometry_refit_balanced": {
            "dimension": selected.get("dimension", {}), "center": selected.get("center", {}),
            "depth": selected.get("depth", {}), "yaw": selected.get("yaw", {}),
        },
    }


def compare_and_select_v51_variant(
    config: Dict[str, Any], variants: Sequence[str] = VARIANT_NAMES,
) -> Dict[str, Any]:
    """Apply hard format/identity gates and the strict fit-source gate."""
    root = output_root(config)
    corrections = read_json(root / "learned_corrections" / "selected_corrections.json")
    rows: List[Dict[str, Any]] = []
    for variant in variants:
        directory = variant_root(config, variant)
        rows.append(_variant_row(
            variant, read_json(directory / "geometry_summary.json"),
            read_json(directory / "validation_summary.json"),
            read_json(directory / "applied_corrections_summary.json"), corrections, config,
        ))
    selected, verdict, reasons = _select_variant(rows, corrections)
    comparison = {
        "selected_variant": selected, "verdict": verdict, "reasons": reasons,
        "fit_source": corrections.get("fit_source"), "variants": rows,
        "selected_corrections": corrections, "test_gt_depth_used": False,
        "camera_specific_test_calibration": "camera_specific_calibration_not_applied",
    }
    write_json(root / "comparison" / "v51_geometry_calibration_summary.json", comparison)
    write_csv(root / "comparison" / "v51_geometry_calibration_summary.csv", rows)
    write_json(root / "comparison" / "selected_variant.json", {"selected_variant": selected, "verdict": verdict, "reasons": reasons})
    write_json(root / "comparison" / "verdict.json", {"label": verdict, "selected_variant": selected, "reasons": reasons})
    _write_baseline_deltas(config, selected, rows)
    return comparison


def _variant_row(
    variant: str, metrics: Dict[str, Any], validation: Dict[str, Any], applied: Dict[str, Any],
    corrections: Dict[str, Any], config: Dict[str, Any],
) -> Dict[str, Any]:
    checks = validation.get("checks", {})
    identity = validation.get("identity_preservation", {})
    failures: List[str] = []
    if validation.get("status") != "ok" or int(validation.get("num_errors", 1)) != 0:
        failures.append("validation_errors")
    for key in ["same_row_count_as_input", "same_unique_track_count_as_input", "identical_row_keys_as_input"]:
        if not identity.get(key, False):
            failures.append(key)
    for key, check_name in [
        ("duplicate_key_count", "duplicate_keys"), ("nan_or_inf_values", "nan_inf"),
        ("non_positive_dimensions", "non_positive_dimensions"), ("rounding_issues", "rounding_issues"),
    ]:
        if int(checks.get(key, 0)) != 0:
            failures.append(check_name)
    expected_scenes = set(str(value) for value in config.get("official_track1", {}).get("valid_scene_ids", []))
    expected_classes = set(str(value) for value in config.get("official_track1", {}).get("valid_class_ids", []))
    if set(metrics.get("scene_distribution", {}).keys()) != expected_scenes:
        failures.append("scene_set_changed")
    if set(metrics.get("class_distribution", {}).keys()) != expected_classes:
        failures.append("class_set_changed")
    risk: List[str] = []
    limits = config.get("selection", {})
    for metric, limit, label in [
        ("mean_position_change_m", "max_position_change_mean_m", "mean_position_change_above_limit"),
        ("p95_position_change_m", "max_position_change_p95_m", "p95_position_change_above_limit"),
        ("max_position_change_m", "max_position_change_max_m", "max_position_change_above_limit"),
        ("p95_dimension_change_ratio", "max_dimension_change_ratio", "dimension_change_ratio_above_limit"),
    ]:
        if _above(metrics.get(metric), limits.get(limit)):
            risk.append(label)
    components = _variant_components(variant)
    counts = {component: sum(1 for item in corrections.get(component, {}).values() if item.get("selected")) for component in components}
    return {
        "variant": variant, "hard_valid": not failures, "hard_failures": failures,
        "quality_risk": bool(risk), "risk_reasons": risk,
        "has_selected_correction": sum(counts.values()) > 0, "selected_correction_counts": counts,
        "rows": metrics.get("rows"), "unique_tracks": metrics.get("unique_tracks"),
        "validation_errors": validation.get("num_errors"), "duplicate_keys": checks.get("duplicate_key_count"),
        "nan_inf": checks.get("nan_or_inf_values"), "non_positive_dimensions": checks.get("non_positive_dimensions"),
        "rounding_issues": checks.get("rounding_issues"), "scene_distribution": metrics.get("scene_distribution"),
        "class_distribution": metrics.get("class_distribution"),
        "mean_position_change_m": metrics.get("mean_position_change_m"),
        "p95_position_change_m": metrics.get("p95_position_change_m"),
        "max_position_change_m": metrics.get("max_position_change_m"),
        "mean_dimension_change_ratio": metrics.get("mean_dimension_change_ratio"),
        "p95_dimension_change_ratio": metrics.get("p95_dimension_change_ratio"),
        "yaw_changed_count": metrics.get("yaw_changed_count"), "yaw_change_mean": metrics.get("yaw_change_mean"),
        "applied_rows_by_component": applied.get("applied_rows_by_component"),
    }


def _select_variant(rows: Sequence[Dict[str, Any]], corrections: Dict[str, Any]) -> Tuple[Any, str, List[str]]:
    if corrections.get("fit_source") != "fit_train":
        return None, "v51_invalid_missing_fit_train_sources", ["fit_source_is_not_fit_train", "internal_holdout_fallback_forbidden"]
    valid = [row for row in rows if row.get("hard_valid")]
    if not valid:
        return None, "v51_invalid_fix_required", ["no_variant_passed_hard_validation"]
    useful = [row for row in valid if row.get("has_selected_correction")]
    if not useful:
        return None, "v51_no_real_improvement_over_v5", ["no_correction_improved_both_internal_holdout_and_official_val"]
    safe = [row for row in useful if not row.get("quality_risk")]
    if not safe:
        return None, "v51_valid_but_quality_risk", ["all_useful_variants_exceeded_test_change_limits"]
    components = [name for name in ["dimension", "center", "yaw"] if any(item.get("selected") for item in corrections.get(name, {}).values())]
    if len(components) > 1 and any(row["variant"] == "v51_geometry_refit_balanced" for row in safe):
        return "v51_geometry_refit_balanced", "v51_geometry_calibration_refit_ready_for_upload", ["multiple_fit_train_corrections_improved_holdout_and_official_val"]
    if components == ["center"]:
        return "v51_center_bias_refit", "v51_center_bias_refit_ready_for_upload", ["center_refit_passed_train_holdout_val_and_test_safety_gates"]
    if components == ["dimension"]:
        return "v51_dimension_scale_refit", "v51_dimension_only_refit_ready_for_upload", ["dimension_refit_passed_train_holdout_val_and_test_safety_gates"]
    preferred = "v51_yaw_bias_refit" if components == ["yaw"] else "v51_geometry_refit_balanced"
    if any(row["variant"] == preferred for row in safe):
        return preferred, "v51_geometry_calibration_refit_ready_for_upload", ["fit_train_refit_passed_holdout_official_val_and_test_safety_gates"]
    return None, "v51_valid_but_small_gain", ["no_safe_preferred_variant"]


def _write_corrections(
    root: Path, learned: Dict[str, Any], selected: Dict[str, Any], rejected: List[Dict[str, Any]],
) -> None:
    directory = root / "learned_corrections"
    write_json(directory / "dimension_scale_refit.json", learned["dimension"])
    write_json(directory / "center_bias_refit.json", learned["center"])
    write_json(directory / "depth_scale_refit.json", learned["depth"])
    write_json(directory / "yaw_bias_refit.json", learned["yaw"])
    write_json(directory / "selected_corrections.json", selected)
    write_json(directory / "rejected_corrections.json", {"corrections": rejected})
    write_json(directory / "correction_source_summary.json", {
        "fit_source": selected.get("fit_source"), "fallback_used": False,
        "fit_scenes": ["Warehouse_%03d" % index for index in range(14)],
        "holdout_scenes": ["Warehouse_%03d" % index for index in range(14, 20)],
        "official_val_scenes": ["Warehouse_%03d" % index for index in range(20, 23)],
    })


def _write_diagnostics(
    root: Path, phases: Dict[str, List[Dict[str, Any]]], selected: Dict[str, Any],
) -> Dict[str, Any]:
    sets = correction_sets(selected)
    names = {
        "fit_train": "fit_train_before_after.csv",
        "internal_holdout": "internal_holdout_before_after.csv",
        "official_val": "official_val_before_after.csv",
    }
    output: Dict[str, Any] = {}
    for phase, filename in names.items():
        rows = before_after_rows(phases.get(phase, []), sets)
        write_csv(root / "validation_diagnostics" / filename, rows)
        output[phase] = {"samples": len(phases.get(phase, [])), "diagnostic_rows": len(rows)}
    all_rows = [row for values in phases.values() for row in values]
    per_class: List[Dict[str, Any]] = []
    for class_id in sorted(set(str(int(row["official_class_id"])) for row in all_rows), key=int):
        for row in before_after_rows(_class_rows(all_rows, class_id), sets):
            row["official_class_id"] = int(class_id)
            per_class.append(row)
    write_csv(root / "validation_diagnostics" / "per_class_before_after.csv", per_class)
    per_scene: List[Dict[str, Any]] = []
    for scene in sorted(set(str(row["scene_name"]) for row in all_rows)):
        for row in before_after_rows([item for item in all_rows if str(item["scene_name"]) == scene], sets):
            row["scene_name"] = scene
            per_scene.append(row)
    write_csv(root / "validation_diagnostics" / "per_scene_before_after.csv", per_scene)
    return output


def _write_baseline_deltas(config: Dict[str, Any], selected: Any, variant_rows: List[Dict[str, Any]]) -> None:
    root = output_root(config)
    candidate_path = variant_root(config, str(selected)) / "track1.txt" if selected else None
    candidate = read_geometry_rows(candidate_path, progress=False) if candidate_path and candidate_path.is_file() else []
    for key, filename in [("v5_track1", "metric_deltas_vs_v5.csv"), ("v4_track1", "metric_deltas_vs_v4.csv"), ("v31_track1", "metric_deltas_vs_v31.csv")]:
        path = Path(str(config.get("paths", {}).get(key, "")))
        baseline = read_geometry_rows(path, progress=False) if path.is_file() else []
        rows = _comparison_rows(baseline, candidate, key) if baseline and candidate else [{"baseline": key, "status": "not_available"}]
        write_csv(root / "comparison" / filename, rows)
        if key == "v5_track1":
            write_csv(root / "validation_diagnostics" / "v5_vs_v51_diagnostics.csv", rows)


def _comparison_rows(baseline: Sequence[Any], candidate: Sequence[Any], name: str) -> List[Dict[str, Any]]:
    summary = summarize_test_changes(baseline, candidate)
    return [{"baseline": name, "metric": key, "candidate_value": value, "status": "row_identity_aligned"} for key, value in summary.items()]


def _single_component(component: str, class_id: str, item: Dict[str, Any]) -> Dict[str, Any]:
    values = {"dimension": {}, "center": {}, "depth": {}, "yaw": {}}
    values[component][class_id] = dict(item, selected=True)
    return values


def _evaluation(rows: Sequence[Dict[str, Any]], corrections: Dict[str, Any], metric: str) -> Dict[str, Any]:
    before = summarize_match_rows(rows).get(metric)
    after = evaluate_corrections(rows, corrections).get(metric)
    return {"samples": len(rows), "metric": metric, "before": before, "after": after, "delta": _delta(after, before)}


def _read_numeric_rows(path: Path) -> List[Dict[str, Any]]:
    """Read one phase incrementally and retain only fields used by fitting."""
    numeric = {
        "official_class_id", "internal_class_id", "frame_id", "pred_x", "pred_y", "pred_z",
        "gt_x", "gt_y", "gt_z", "pred_width", "pred_length", "pred_height",
        "gt_width", "gt_length", "gt_height", "pred_yaw", "gt_yaw", "pred_distance",
        "gt_distance", "center_error_before", "dimension_error_before", "yaw_error_before",
        "depth_error_before", "iou3d_proxy_before",
    }
    retained = numeric | {"scene_name", "coordinate_frame"}
    output: List[Dict[str, Any]] = []
    if not path.is_file():
        return output
    with path.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            item: Dict[str, Any] = {}
            valid = True
            for key in retained:
                value = row.get(key)
                if key not in numeric:
                    item[key] = value
                    continue
                if value in (None, ""):
                    continue
                try:
                    item[key] = float(value)
                except (TypeError, ValueError):
                    valid = False
                    break
            if valid:
                output.append(item)
    return output


def _class_rows(rows: Sequence[Dict[str, Any]], class_id: str) -> List[Dict[str, Any]]:
    return [row for row in rows if str(int(float(row["official_class_id"]))) == str(class_id)]


def _coordinate_safe_rows(rows: Sequence[Dict[str, Any]], config: Dict[str, Any]) -> List[Dict[str, Any]]:
    rules = config.get("matching", {})
    allowed = set(str(value).lower() for value in rules.get("center_yaw_coordinate_frames", ["world", "global"]))
    allow_unknown = bool(rules.get("allow_unknown_coordinate_frame_for_center_yaw", False))
    return [row for row in rows if str(row.get("coordinate_frame", "unknown")).lower() in allowed or (allow_unknown and str(row.get("coordinate_frame", "unknown")).lower() == "unknown")]


def _improves(evaluation: Dict[str, Any]) -> bool:
    delta = evaluation.get("delta")
    return delta is not None and float(delta) < -1e-9


def _reject_reason(
    fit_source: str, enough: bool, stable: bool, holdout: Sequence[Any], official_val: Sequence[Any],
    holdout_improved: bool, val_improved: bool,
) -> str:
    if fit_source != "fit_train":
        return "missing_fit_train_sources"
    if not enough:
        return "insufficient_fit_train_samples"
    if not stable:
        return "fit_train_variability_too_high"
    if not holdout:
        return "missing_internal_holdout_samples"
    if not official_val:
        return "missing_official_val_samples"
    if not holdout_improved:
        return "internal_holdout_not_improved"
    if not val_improved:
        return "official_val_not_improved"
    return "not_selected"


def _variant_components(variant: str) -> List[str]:
    mapping = {
        "v51_dimension_scale_refit": ["dimension"], "v51_center_bias_refit": ["center"],
        "v51_depth_scale_refit": ["depth"], "v51_yaw_bias_refit": ["yaw"],
    }
    return mapping.get(variant, ["dimension", "center", "depth", "yaw"])


def _above(value: Any, limit: Any) -> bool:
    try:
        return float(value) > float(limit)
    except (TypeError, ValueError):
        return False


def _delta(after: Any, before: Any) -> Any:
    try:
        return float(after) - float(before)
    except (TypeError, ValueError):
        return None
