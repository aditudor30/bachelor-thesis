"""Compare V4 geometry variants and select a conservative upload candidate."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from deep_oc_sort_3d.v4_geometry_refinement.geometry_refinement_config import VARIANT_NAMES, output_root, variant_root
from deep_oc_sort_3d.v4_geometry_refinement.track1_geometry_io import read_geometry_rows, read_json, unique_track_count, write_csv, write_json


PROXY_METRICS = [
    "step_p95",
    "suspect_track_count",
    "suspect_point_count",
    "dimension_variance_mean",
    "yaw_jump_count",
]


def compare_and_select_geometry_variant(
    config: Dict[str, Any],
    variants: Sequence[str] = VARIANT_NAMES,
) -> Dict[str, Any]:
    """Apply hard safety gates, compare local proxies and select one variant."""
    root = output_root(config)
    baseline = read_json(root / "audit" / "v31_geometry_audit.json")
    rows: List[Dict[str, Any]] = []
    for variant in variants:
        metrics = read_json(variant_root(config, variant) / "geometry_summary.json")
        validation = read_json(variant_root(config, variant) / "validation_summary.json")
        rows.append(_variant_comparison_row(variant, metrics, validation, baseline, config))

    selected, verdict, reasons = _select(rows, config)
    comparison = {
        "baseline_variant": "v3_coverage_extended_official",
        "baseline_metrics": baseline,
        "selected_variant": selected,
        "verdict": verdict,
        "reasons": reasons,
        "variants": rows,
        "v2_v3_reference": _reference_rows(config),
        "local_proxy_warning": (
            "Geometry metrics are local consistency proxies only. Test scenes 023-027 have no GT, "
            "so purity, false-merge and true 3D accuracy deltas are unavailable."
        ),
    }
    write_json(root / "comparison" / "v4_geometry_refinement_summary.json", comparison)
    write_csv(root / "comparison" / "v4_geometry_refinement_summary.csv", rows)
    write_csv(root / "comparison" / "metric_deltas_vs_v31.csv", _delta_rows(rows, baseline, "v31"))
    references = comparison.get("v2_v3_reference", [])
    write_csv(root / "comparison" / "metric_deltas_vs_v3.csv", _reference_delta_rows(rows, references, "v3_gap_aware_soft_official"))
    write_csv(root / "comparison" / "metric_deltas_vs_v2.csv", _reference_delta_rows(rows, references, "v2_current_official"))
    write_json(root / "comparison" / "selected_variant.json", {
        "selected_variant": selected,
        "verdict": verdict,
        "reasons": reasons,
        "selection_policy": "hard safety gates followed by conservative local-proxy improvement scoring",
    })
    write_json(root / "comparison" / "verdict.json", {
        "label": verdict,
        "reasons": reasons,
        "selected_variant": selected,
    })
    return comparison


def _delta_rows(rows: Sequence[Dict[str, Any]], baseline: Dict[str, Any], baseline_name: str) -> List[Dict[str, Any]]:
    metrics = ["rows", "unique_tracks"] + PROXY_METRICS + [
        "jump_count", "z_outlier_count", "mean_position_change_m", "p95_position_change_m",
        "max_position_change_m", "dimension_change_count", "yaw_changed_count",
    ]
    output: List[Dict[str, Any]] = []
    for row in rows:
        for metric in metrics:
            baseline_value = _number(baseline.get(metric))
            candidate_value = _number(row.get(metric))
            output.append({
                "variant": row.get("variant"), "baseline": baseline_name, "metric": metric,
                "baseline_value": baseline_value, "candidate_value": candidate_value,
                "delta": None if baseline_value is None or candidate_value is None else candidate_value - baseline_value,
                "status": "ok" if baseline_value is not None and candidate_value is not None else "not_available",
            })
    return output


def _reference_delta_rows(rows: Sequence[Dict[str, Any]], references: Any, reference_name: str) -> List[Dict[str, Any]]:
    reference = next((item for item in references if item.get("variant") == reference_name), {})
    output: List[Dict[str, Any]] = []
    for row in rows:
        for metric in ["rows", "unique_tracks"] + PROXY_METRICS:
            reference_value = _number(reference.get(metric))
            candidate_value = _number(row.get(metric))
            output.append({
                "variant": row.get("variant"), "baseline": reference_name, "metric": metric,
                "baseline_value": reference_value, "candidate_value": candidate_value,
                "delta": None if reference_value is None or candidate_value is None else candidate_value - reference_value,
                "status": "ok" if reference_value is not None and candidate_value is not None else "not_available",
                "note": "Geometry proxies are intentionally unavailable for V2/V3 because their identity sets differ." if metric in PROXY_METRICS else "",
            })
    return output


def _variant_comparison_row(
    variant: str,
    metrics: Dict[str, Any],
    validation: Dict[str, Any],
    baseline: Dict[str, Any],
    config: Dict[str, Any],
) -> Dict[str, Any]:
    identity = validation.get("identity_preservation", {})
    selection = config.get("selection", {})
    checks = validation.get("checks", {})
    hard_failures: List[str] = []
    if validation.get("status") != "ok" or int(validation.get("num_errors", 1)) != 0:
        hard_failures.append("validation_errors")
    if selection.get("require_same_row_count_as_v31", True) and not identity.get("same_row_count_as_v31", False):
        hard_failures.append("row_count_changed")
    if selection.get("require_same_unique_track_count_as_v31", True) and not identity.get("same_unique_track_count_as_v31", False):
        hard_failures.append("unique_track_count_changed")
    if selection.get("require_identical_row_keys_as_v31", True) and not identity.get("identical_row_keys_as_v31", False):
        hard_failures.append("row_keys_changed")
    if int(checks.get("duplicate_key_count", 0)) > int(selection.get("max_duplicate_keys_allowed", 0)):
        hard_failures.append("duplicate_keys")
    if int(checks.get("nan_or_inf_values", 0)) > int(selection.get("max_nan_inf_allowed", 0)):
        hard_failures.append("nan_or_inf")
    if int(checks.get("non_positive_dimensions", 0)) > int(selection.get("max_non_positive_dimensions_allowed", 0)):
        hard_failures.append("non_positive_dimensions")
    if int(checks.get("rounding_issues", 0)) > int(selection.get("max_rounding_issues_allowed", 0)):
        hard_failures.append("rounding_issues")
    expected_scenes = set(str(value) for value in config.get("official_track1", {}).get("valid_scene_ids", []))
    actual_scenes = set(str(value) for value in metrics.get("scene_distribution", {}).keys())
    if expected_scenes and actual_scenes != expected_scenes:
        hard_failures.append("official_scene_set_changed")
    expected_classes = set(str(value) for value in config.get("official_track1", {}).get("valid_class_ids", []))
    actual_classes = set(str(value) for value in metrics.get("class_distribution", {}).keys())
    if expected_classes and actual_classes != expected_classes:
        hard_failures.append("official_class_set_changed")

    aggressive_reasons = _aggressive_reasons(metrics, config)
    improvement_count = 0
    regression_count = 0
    deltas: Dict[str, Any] = {}
    for metric in PROXY_METRICS:
        candidate_value = _number(metrics.get(metric))
        baseline_value = _number(baseline.get(metric))
        delta = None if candidate_value is None or baseline_value is None else candidate_value - baseline_value
        deltas[metric + "_delta_vs_v31"] = delta
        if delta is not None and delta < -1e-9:
            improvement_count += 1
        elif delta is not None and delta > 1e-9:
            regression_count += 1
    score = float(improvement_count * 10 - regression_count * 4)
    step_base = _number(baseline.get("step_p95"))
    step_value = _number(metrics.get("step_p95"))
    if step_base and step_value is not None:
        score += max(-5.0, min(5.0, 5.0 * (step_base - step_value) / step_base))
    if variant == "v4_geometry_refined_balanced":
        score += 0.25
    return {
        "variant": variant,
        "status": "ok" if metrics else "missing_metrics",
        "hard_valid": len(hard_failures) == 0,
        "hard_failures": hard_failures,
        "too_aggressive": bool(aggressive_reasons),
        "aggressive_reasons": aggressive_reasons,
        "proxy_improvement_count": improvement_count,
        "proxy_regression_count": regression_count,
        "selection_score": score,
        "rows": metrics.get("rows"),
        "unique_tracks": metrics.get("unique_tracks"),
        "validation_errors": validation.get("num_errors"),
        "duplicate_keys": checks.get("duplicate_key_count"),
        "nan_inf": checks.get("nan_or_inf_values"),
        "non_positive_dimensions": checks.get("non_positive_dimensions"),
        "rounding_issues": checks.get("rounding_issues"),
        "step_p95": metrics.get("step_p95"),
        "suspect_track_count": metrics.get("suspect_track_count"),
        "suspect_point_count": metrics.get("suspect_point_count"),
        "dimension_variance_mean": metrics.get("dimension_variance_mean"),
        "yaw_jump_count": metrics.get("yaw_jump_count"),
        "points_repaired": metrics.get("points_repaired"),
        "points_changed": metrics.get("points_changed"),
        "jump_count": metrics.get("jump_count"),
        "z_outlier_count": metrics.get("z_outlier_count"),
        "mean_position_change_m": metrics.get("mean_position_change_m"),
        "p95_position_change_m": metrics.get("p95_position_change_m"),
        "max_position_change_m": metrics.get("max_position_change_m"),
        "dimension_change_ratio_p95": metrics.get("dimension_change_ratio_p95"),
        "dimension_change_ratio_max": metrics.get("dimension_change_ratio_max"),
        "dimension_change_count": metrics.get("dimension_change_count"),
        "yaw_changed_count": metrics.get("yaw_changed_count"),
        "yaw_change_p95_rad": metrics.get("yaw_change_p95_rad"),
        "scene_distribution": metrics.get("scene_distribution"),
        "class_distribution": metrics.get("class_distribution"),
        **deltas,
    }


def _aggressive_reasons(metrics: Dict[str, Any], config: Dict[str, Any]) -> List[str]:
    selection = config.get("selection", {})
    checks = [
        ("mean_position_change_m", "max_mean_position_change_m"),
        ("p95_position_change_m", "max_p95_position_change_m"),
        ("max_position_change_m", "max_position_change_m"),
    ]
    reasons = []
    for metric, limit_key in checks:
        value = _number(metrics.get(metric))
        limit = _number(selection.get(limit_key))
        if value is not None and limit is not None and value > limit:
            reasons.append("%s_above_limit" % metric)
    dimension_limit = _number(config.get("dimension_consistency", {}).get("max_dimension_change_ratio_warning", 0.50))
    dimension_value = _number(metrics.get("dimension_change_ratio_p95"))
    if dimension_value is not None and dimension_limit is not None and dimension_value > dimension_limit:
        reasons.append("dimension_change_ratio_p95_above_warning")
    return reasons


def _select(rows: Sequence[Dict[str, Any]], config: Dict[str, Any]) -> Tuple[Optional[str], str, List[str]]:
    existing = [row for row in rows if row.get("status") == "ok"]
    if not existing or all(not bool(row.get("hard_valid")) for row in existing):
        return None, "v4_geometry_refinement_invalid_fix_required", ["no_variant_passed_hard_safety_gates"]
    valid = [row for row in existing if row.get("hard_valid")]
    conservative = [row for row in valid if not row.get("too_aggressive")]
    if not conservative:
        return None, "v4_geometry_refinement_valid_but_too_aggressive", ["all_valid_variants_exceeded_change_limits"]
    minimum = int(config.get("selection", {}).get("minimum_proxy_improvements", 1))
    useful = [row for row in conservative if int(row.get("proxy_improvement_count", 0)) >= minimum]
    if not useful:
        return None, "v4_geometry_refinement_valid_but_small_gain", ["no_conservative_variant_met_minimum_proxy_improvements"]
    selected = sorted(useful, key=lambda row: (float(row.get("selection_score", 0.0)), row.get("variant") == "v4_geometry_refined_balanced"), reverse=True)[0]
    if float(selected.get("selection_score", 0.0)) <= 0.0:
        return None, "v31_remains_better_candidate", ["best_v4_proxy_score_did_not_improve_over_v31"]
    return str(selected.get("variant")), "v4_geometry_refinement_ready_for_upload", [
        "identity_and_validation_gates_passed",
        "local_geometry_proxies_improved_within_conservative_change_limits",
    ]


def _reference_rows(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    output = []
    for name, key in [("v2_current_official", "v2_official_track1"), ("v3_gap_aware_soft_official", "v3_official_track1")]:
        path = Path(str(config.get("paths", {}).get(key, "")))
        rows = read_geometry_rows(path, progress=False) if path.is_file() else []
        output.append({
            "variant": name,
            "track1_path": str(path),
            "available": path.is_file(),
            "rows": len(rows) if rows else None,
            "unique_tracks": unique_track_count(rows) if rows else None,
            "geometry_proxy_metrics": "not_computed_not_same_identity_set",
        })
    return output


def _number(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None
