"""Fit, validate and select only robust V5 geometry corrections."""

from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

from deep_oc_sort_3d.v5_geometry_calibration.center_bias_calibrator import fit_center_biases
from deep_oc_sort_3d.v5_geometry_calibration.depth_scale_calibrator import fit_depth_scales
from deep_oc_sort_3d.v5_geometry_calibration.dimension_calibrator import fit_dimension_scales
from deep_oc_sort_3d.v5_geometry_calibration.geometry_calibration_config import VARIANT_NAMES, input_track1_path, input_variant_name, output_root, variant_root
from deep_oc_sort_3d.v5_geometry_calibration.geometry_calibration_io import read_csv, read_geometry_rows, read_json, unique_track_count, write_csv, write_json
from deep_oc_sort_3d.v5_geometry_calibration.geometry_calibration_metrics import before_after_rows, evaluate_corrections, summarize_match_rows
from deep_oc_sort_3d.v5_geometry_calibration.yaw_calibrator import fit_yaw_biases


def fit_and_select_corrections(config: Dict[str, Any]) -> Dict[str, Any]:
    """Fit on configured train phase and gate corrections on holdout/official_val."""
    root = output_root(config)
    rows = _numeric_rows(read_csv(root / "calibration_dataset" / "calibration_matches.csv"))
    phases = _split_rows(rows)
    fit_rows = phases.get("fit_train", [])
    fit_source = "fit_train"
    warnings: List[str] = []
    if not fit_rows:
        fit_rows = phases.get("internal_holdout", [])
        fit_source = "internal_holdout_fallback"
        warnings.append("fit_train observations unavailable; corrections fitted on internal_holdout and gated on official_val")
    world_fit_rows = _coordinate_safe_rows(fit_rows, config)
    if len(world_fit_rows) < len(fit_rows):
        warnings.append("center and yaw calibration excluded rows without an explicit world/global coordinate frame")
    learned = {
        "dimension": fit_dimension_scales(fit_rows, config),
        "center": fit_center_biases(world_fit_rows, config),
        "depth": fit_depth_scales(fit_rows, config),
        "yaw": fit_yaw_biases(world_fit_rows, config),
    }
    evaluation_rows = phases.get("official_val", []) or phases.get("internal_holdout", [])
    holdout_rows = phases.get("internal_holdout", []) if fit_source == "fit_train" else phases.get("official_val", [])
    rejected: List[Dict[str, Any]] = []
    selected: List[Dict[str, Any]] = []
    for component, metric in [("dimension", "dimension_error_mean"), ("center", "center_error_mean"), ("yaw", "yaw_error_mean")]:
        for class_id, item in learned[component].items():
            class_eval = _class_rows(evaluation_rows, class_id)
            class_holdout = _class_rows(holdout_rows, class_id)
            if component in ("center", "yaw"):
                class_eval = _coordinate_safe_rows(class_eval, config)
                class_holdout = _coordinate_safe_rows(class_holdout, config)
            candidate = {"dimension": {}, "center": {}, "depth": {}, "yaw": {}}
            candidate[component][class_id] = dict(item, selected=True)
            evaluation = _component_evaluation(class_eval, candidate, metric)
            holdout = _component_evaluation(class_holdout, candidate, metric)
            eligible_samples = bool(item.get("eligible_by_sample_count"))
            eligible_variability = bool(item.get("eligible_by_variability", True))
            enough = eligible_samples and eligible_variability and bool(class_eval)
            improves_eval = _improves(evaluation)
            improves_holdout = True if not class_holdout else _improves(holdout)
            accepted = enough and improves_eval and improves_holdout
            item["selected"] = accepted
            item["evaluation"] = evaluation
            item["holdout_evaluation"] = holdout
            record = {"component": component, "official_class_id": int(class_id), "accepted": accepted, **item}
            if accepted:
                selected.append(record)
            else:
                record["reject_reason"] = _reject_reason(
                    eligible_samples, eligible_variability, bool(class_eval), improves_eval, improves_holdout,
                )
                rejected.append(record)
    for class_id, item in learned["depth"].items():
        item["selected"] = False
        rejected.append({
            "component": "depth", "official_class_id": int(class_id), "accepted": False,
            "reject_reason": "not_applied_due_to_missing_camera_mapping", **item,
        })
    selected_corrections = {
        "fit_source": fit_source, "warnings": warnings,
        "dimension": learned["dimension"], "center": learned["center"],
        "depth": learned["depth"], "yaw": learned["yaw"],
    }
    _write_corrections(root, learned, selected_corrections, selected, rejected)
    diagnostics = _write_validation_diagnostics(root, phases, learned, selected_corrections)
    verdict = {
        "status": "ok" if rows else "no_calibration_matches",
        "fit_source": fit_source, "selected_count": len(selected), "rejected_count": len(rejected),
        "selected_components": sorted(set(item["component"] for item in selected)),
        "warnings": warnings, "diagnostics": diagnostics,
    }
    write_json(root / "validation_diagnostics" / "calibration_verdict.json", verdict)
    return verdict


def correction_sets(selected: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Build correction dictionaries for all five V5 variants."""
    empty = {"dimension": {}, "center": {}, "depth": {}, "yaw": {}}
    dimension = dict(empty, dimension=selected.get("dimension", {}))
    center = dict(empty, center=selected.get("center", {}))
    depth = dict(empty, depth=selected.get("depth", {}))
    yaw = dict(empty, yaw=selected.get("yaw", {}))
    balanced = {
        "dimension": selected.get("dimension", {}), "center": selected.get("center", {}),
        "depth": selected.get("depth", {}), "yaw": selected.get("yaw", {}),
    }
    return {
        "v5_dimension_scale_calibrated": dimension,
        "v5_center_bias_calibrated": center,
        "v5_depth_scale_calibrated": depth,
        "v5_yaw_bias_calibrated": yaw,
        "v5_geometry_calibrated_balanced": balanced,
    }


def compare_and_select_v5_variant(config: Dict[str, Any], variants: Sequence[str] = VARIANT_NAMES) -> Dict[str, Any]:
    """Select a valid V5 variant using accepted train/val corrections and test safety gates."""
    root = output_root(config)
    corrections = read_json(root / "learned_corrections" / "selected_corrections.json")
    rows: List[Dict[str, Any]] = []
    for variant in variants:
        variant_path = variant_root(config, variant)
        metrics = read_json(variant_path / "geometry_summary.json")
        validation = read_json(variant_path / "validation_summary.json")
        applied = read_json(variant_path / "applied_corrections_summary.json")
        row = _variant_row(variant, metrics, validation, applied, corrections, config)
        rows.append(row)
    selected, verdict, reasons = _select_variant(rows, corrections, config)
    comparison = {
        "selected_variant": selected, "verdict": verdict, "reasons": reasons,
        "variants": rows, "selected_corrections": corrections,
        "test_gt_depth_used": False,
        "camera_specific_test_calibration": "not_applied_due_to_missing_camera_mapping",
    }
    write_json(root / "comparison" / "v5_geometry_calibration_summary.json", comparison)
    write_csv(root / "comparison" / "v5_geometry_calibration_summary.csv", rows)
    input_rows = read_geometry_rows(input_track1_path(config), progress=False)
    v31_path = Path(str(config.get("paths", {}).get("v31_track1", "")))
    v31_rows = read_geometry_rows(v31_path, progress=False) if v31_path.is_file() else []
    write_csv(
        root / "comparison" / "metric_deltas_vs_v4.csv",
        _metric_delta_rows(rows, "immutable_input", len(input_rows), unique_track_count(input_rows), True),
    )
    write_csv(
        root / "comparison" / "metric_deltas_vs_v31.csv",
        _metric_delta_rows(rows, "v31_reference", len(v31_rows), unique_track_count(v31_rows), input_track1_path(config) == v31_path),
    )
    write_csv(root / "comparison" / "train_val_calibration_report.csv", _train_val_report(corrections))
    write_json(root / "comparison" / "selected_variant.json", {"selected_variant": selected, "verdict": verdict, "reasons": reasons})
    write_json(root / "comparison" / "verdict.json", {"label": verdict, "selected_variant": selected, "reasons": reasons})
    return comparison


def _component_evaluation(rows: Sequence[Dict[str, Any]], correction: Dict[str, Any], metric: str) -> Dict[str, Any]:
    before = summarize_match_rows(rows).get(metric)
    after = evaluate_corrections(rows, correction).get(metric)
    return {"samples": len(rows), "metric": metric, "before": before, "after": after, "delta": _delta(after, before)}


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
    if int(checks.get("duplicate_key_count", 0)) != 0:
        failures.append("duplicate_keys")
    if int(checks.get("nan_or_inf_values", 0)) != 0:
        failures.append("nan_inf")
    if int(checks.get("non_positive_dimensions", 0)) != 0:
        failures.append("non_positive_dimensions")
    if int(checks.get("rounding_issues", 0)) != 0:
        failures.append("rounding_issues")
    expected_scenes = set(str(value) for value in config.get("official_track1", {}).get("valid_scene_ids", []))
    expected_classes = set(str(value) for value in config.get("official_track1", {}).get("valid_class_ids", []))
    if set(metrics.get("scene_distribution", {}).keys()) != expected_scenes:
        failures.append("scene_set_changed")
    if set(metrics.get("class_distribution", {}).keys()) != expected_classes:
        failures.append("class_set_changed")
    risk: List[str] = []
    selection = config.get("selection", {})
    if _above(metrics.get("mean_position_change_m"), selection.get("max_position_change_mean_m", 1.0)):
        risk.append("mean_position_change_above_limit")
    if _above(metrics.get("p95_position_change_m"), selection.get("max_position_change_p95_m", 3.0)):
        risk.append("p95_position_change_above_limit")
    if _above(metrics.get("p95_dimension_change_ratio"), selection.get("max_dimension_change_ratio", 0.20)):
        risk.append("dimension_change_ratio_above_limit")
    components = _variant_components(variant)
    selected_counts = {component: sum(1 for item in corrections.get(component, {}).values() if item.get("selected")) for component in components}
    useful = sum(selected_counts.values()) > 0
    return {
        "variant": variant, "hard_valid": not failures, "hard_failures": failures,
        "quality_risk": bool(risk), "risk_reasons": risk, "has_selected_correction": useful,
        "selected_correction_counts": selected_counts,
        "rows": metrics.get("rows"), "unique_tracks": metrics.get("unique_tracks"),
        "validation_errors": validation.get("num_errors"), "duplicate_keys": checks.get("duplicate_key_count"),
        "nan_inf": checks.get("nan_or_inf_values"), "non_positive_dimensions": checks.get("non_positive_dimensions"),
        "rounding_issues": checks.get("rounding_issues"), "scene_distribution": metrics.get("scene_distribution"),
        "class_distribution": metrics.get("class_distribution"),
        "mean_position_change_m": metrics.get("mean_position_change_m"),
        "p95_position_change_m": metrics.get("p95_position_change_m"), "max_position_change_m": metrics.get("max_position_change_m"),
        "mean_dimension_change_ratio": metrics.get("mean_dimension_change_ratio"),
        "p95_dimension_change_ratio": metrics.get("p95_dimension_change_ratio"),
        "yaw_changed_count": metrics.get("yaw_changed_count"), "yaw_change_mean": metrics.get("yaw_change_mean"),
        "applied_rows_by_component": applied.get("applied_rows_by_component"),
    }


def _select_variant(
    rows: Sequence[Dict[str, Any]], corrections: Dict[str, Any], config: Dict[str, Any],
) -> Tuple[Any, str, List[str]]:
    valid = [row for row in rows if row.get("hard_valid")]
    if not valid:
        return None, "v5_geometry_calibration_invalid_fix_required", ["no_variant_passed_hard_validation"]
    safe = [row for row in valid if not row.get("quality_risk") and row.get("has_selected_correction")]
    if not safe:
        if any(row.get("has_selected_correction") for row in valid):
            return None, "v5_geometry_calibration_valid_but_quality_risk", ["all_useful_variants_exceeded_test_change_limits"]
        fallback_verdict = "v4_remains_better_candidate" if input_variant_name(config) == "v4_geometry_refined_official" else "v31_remains_better_candidate"
        return None, fallback_verdict, ["no_train_val_correction_passed_selection"]
    selected_components = [component for component in ["dimension", "center", "yaw"] if any(item.get("selected") for item in corrections.get(component, {}).values())]
    if len(selected_components) > 1:
        balanced = next((row for row in safe if row.get("variant") == "v5_geometry_calibrated_balanced"), None)
        if balanced is not None:
            return "v5_geometry_calibrated_balanced", "v5_geometry_calibration_ready_for_upload", ["multiple_train_val_validated_corrections_combined_safely"]
    if "dimension" in selected_components:
        dimension = next((row for row in safe if row.get("variant") == "v5_dimension_scale_calibrated"), None)
        if dimension is not None:
            return "v5_dimension_scale_calibrated", "v5_dimension_only_calibration_ready_for_upload", ["dimension_calibration_improved_train_val_and_passed_test_safety_gates"]
    for component, preferred in [("center", "v5_center_bias_calibrated"), ("yaw", "v5_yaw_bias_calibrated")]:
        if component in selected_components and any(row.get("variant") == preferred for row in safe):
            return preferred, "v5_geometry_calibration_ready_for_upload", ["train_val_validated_correction_passed_test_safety_gates"]
    return None, "v5_geometry_calibration_valid_but_small_gain", ["no_safe_variant_with_measurable_selected_correction"]


def _variant_components(variant: str) -> List[str]:
    if variant == "v5_dimension_scale_calibrated":
        return ["dimension"]
    if variant == "v5_center_bias_calibrated":
        return ["center"]
    if variant == "v5_depth_scale_calibrated":
        return ["depth"]
    if variant == "v5_yaw_bias_calibrated":
        return ["yaw"]
    return ["dimension", "center", "depth", "yaw"]


def _metric_delta_rows(
    rows: Sequence[Dict[str, Any]], baseline: str, baseline_rows: int,
    baseline_tracks: int, same_geometry_source: bool,
) -> List[Dict[str, Any]]:
    metrics = [
        "rows", "unique_tracks", "mean_position_change_m", "p95_position_change_m", "max_position_change_m",
        "mean_dimension_change_ratio", "p95_dimension_change_ratio", "yaw_changed_count", "yaw_change_mean",
    ]
    output = []
    for row in rows:
        for metric in metrics:
            candidate = row.get(metric)
            if metric == "rows":
                baseline_value: Any = baseline_rows
                delta = _delta(candidate, baseline_value)
                status = "comparable"
            elif metric == "unique_tracks":
                baseline_value = baseline_tracks
                delta = _delta(candidate, baseline_value)
                status = "comparable"
            elif same_geometry_source:
                baseline_value = 0.0
                delta = candidate
                status = "change_from_immutable_input"
            else:
                baseline_value = None
                delta = None
                status = "not_available_without_row_identity_alignment"
            output.append({
                "variant": row.get("variant"), "baseline": baseline, "metric": metric,
                "candidate_value": candidate, "baseline_value": baseline_value,
                "delta": delta, "status": status,
            })
    return output


def _train_val_report(corrections: Dict[str, Any]) -> List[Dict[str, Any]]:
    output = []
    for component in ["dimension", "center", "depth", "yaw"]:
        for class_id, item in corrections.get(component, {}).items():
            evaluation = item.get("evaluation", {})
            holdout = item.get("holdout_evaluation", {})
            output.append({
                "component": component, "official_class_id": int(class_id), "selected": item.get("selected"),
                "samples_fit": item.get("samples"), "eval_samples": evaluation.get("samples"),
                "eval_metric": evaluation.get("metric"), "eval_before": evaluation.get("before"),
                "eval_after": evaluation.get("after"), "eval_delta": evaluation.get("delta"),
                "holdout_samples": holdout.get("samples"), "holdout_delta": holdout.get("delta"),
            })
    return output


def _above(value: Any, limit: Any) -> bool:
    try:
        return float(value) > float(limit)
    except (TypeError, ValueError):
        return False


def _write_corrections(
    root: Path, learned: Dict[str, Any], selected_corrections: Dict[str, Any],
    selected: List[Dict[str, Any]], rejected: List[Dict[str, Any]],
) -> None:
    directory = root / "learned_corrections"
    write_json(directory / "dimension_scale_corrections.json", learned["dimension"])
    write_json(directory / "center_bias_corrections.json", learned["center"])
    write_json(directory / "depth_scale_corrections.json", learned["depth"])
    write_json(directory / "yaw_bias_corrections.json", learned["yaw"])
    write_json(directory / "selected_corrections.json", selected_corrections)
    write_json(directory / "rejected_corrections.json", {"corrections": rejected})


def _write_validation_diagnostics(
    root: Path, phases: Dict[str, List[Dict[str, Any]]], learned: Dict[str, Any], selected: Dict[str, Any],
) -> Dict[str, Any]:
    sets = correction_sets(selected)
    outputs = {}
    paths = {
        "fit_train": "train_eval_before_after.csv",
        "internal_holdout": "holdout_eval_before_after.csv",
        "official_val": "official_val_eval_before_after.csv",
    }
    for phase, filename in paths.items():
        values = before_after_rows(phases.get(phase, []), sets)
        write_csv(root / "validation_diagnostics" / filename, values)
        outputs[phase] = {"samples": len(phases.get(phase, [])), "rows": len(values)}
    per_class = []
    all_rows = [row for values in phases.values() for row in values]
    for class_id in sorted(set(str(int(row["official_class_id"])) for row in all_rows), key=int):
        values = _class_rows(all_rows, class_id)
        for row in before_after_rows(values, sets):
            row["official_class_id"] = int(class_id)
            per_class.append(row)
    write_csv(root / "validation_diagnostics" / "per_class_before_after.csv", per_class)
    per_scene = []
    for scene in sorted(set(str(row["scene_name"]) for row in all_rows)):
        values = [row for row in all_rows if str(row["scene_name"]) == scene]
        for row in before_after_rows(values, sets):
            row["scene_name"] = scene
            per_scene.append(row)
    write_csv(root / "validation_diagnostics" / "per_scene_before_after.csv", per_scene)
    return outputs


def _numeric_rows(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    numeric = {
        "official_class_id", "internal_class_id", "frame_id", "pred_x", "pred_y", "pred_z", "gt_x", "gt_y", "gt_z",
        "pred_width", "pred_length", "pred_height", "gt_width", "gt_length", "gt_height", "pred_yaw", "gt_yaw",
        "pred_distance", "gt_distance", "center_error_before", "dimension_error_before", "yaw_error_before",
        "depth_error_before", "iou3d_proxy_before",
    }
    output = []
    for row in rows:
        item = dict(row)
        valid = True
        for key in numeric:
            if item.get(key) in (None, ""):
                continue
            try:
                item[key] = float(item[key])
            except (TypeError, ValueError):
                valid = False
                break
        if valid:
            output.append(item)
    return output


def _split_rows(rows: Sequence[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    output: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        output[str(row.get("phase", "unknown"))].append(row)
    return output


def _class_rows(rows: Sequence[Dict[str, Any]], class_id: str) -> List[Dict[str, Any]]:
    return [row for row in rows if str(int(float(row["official_class_id"]))) == str(class_id)]


def _coordinate_safe_rows(rows: Sequence[Dict[str, Any]], config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Keep rows whose predictions explicitly use the world/global coordinate frame."""
    matching = config.get("matching", {})
    allowed = set(str(value).lower() for value in matching.get("center_yaw_coordinate_frames", ["world", "global"]))
    allow_unknown = bool(matching.get("allow_unknown_coordinate_frame_for_center_yaw", False))
    output = []
    for row in rows:
        coordinate_frame = str(row.get("coordinate_frame") or "unknown").lower()
        if coordinate_frame in allowed or (allow_unknown and coordinate_frame == "unknown"):
            output.append(row)
    return output


def _improves(evaluation: Dict[str, Any]) -> bool:
    delta = evaluation.get("delta")
    return delta is not None and float(delta) < -1e-9


def _reject_reason(
    eligible_samples: bool, eligible_variability: bool, has_evaluation_rows: bool,
    improves_eval: bool, improves_holdout: bool,
) -> str:
    if not eligible_samples:
        return "insufficient_samples"
    if not eligible_variability:
        return "error_variability_too_high"
    if not has_evaluation_rows:
        return "no_official_val_or_primary_evaluation_samples"
    if not improves_eval:
        return "official_val_or_primary_evaluation_not_improved"
    if not improves_holdout:
        return "holdout_not_improved"
    return "not_selected"


def _delta(after: Any, before: Any) -> Any:
    try:
        return float(after) - float(before)
    except (TypeError, ValueError):
        return None
