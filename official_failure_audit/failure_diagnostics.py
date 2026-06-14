"""Interpret Step 23A sweep results into actionable failure diagnoses."""

from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

from deep_oc_sort_3d.official_failure_audit.failure_io import write_json


def write_failure_diagnostics(
    original: Dict[str, Any], sweep: Dict[str, Any], source_summary: Dict[str, Any], output_root: Path,
) -> Dict[str, Any]:
    directory = output_root / "diagnostics"
    individual = sweep.get("individual", [])
    diagnoses = {
        "axis": _category_diagnosis("axis", original, individual),
        "center": _category_diagnosis("center", original, individual),
        "dimension": _category_diagnosis("dimension", original, individual),
        "yaw": _category_diagnosis("yaw", original, individual),
        "frame": _category_diagnosis("frame", original, individual),
        "class": _category_diagnosis("class", original, individual),
    }
    write_json(directory / "coordinate_frame_diagnosis.json", diagnoses["axis"])
    write_json(directory / "center_convention_diagnosis.json", diagnoses["center"])
    write_json(directory / "dimension_order_diagnosis.json", diagnoses["dimension"])
    write_json(directory / "yaw_diagnosis.json", diagnoses["yaw"])
    write_json(directory / "frame_offset_diagnosis.json", diagnoses["frame"])
    write_json(directory / "class_mapping_diagnosis.json", diagnoses["class"])

    verdict, causes = _verdict(diagnoses, source_summary)
    best = sweep.get("best", {})
    likely = {
        "verdict": verdict, "likely_causes": causes, "original": original,
        "best_hypothesis": best, "category_diagnoses": diagnoses,
        "official_metric_symptoms": {
            "DetA": "Near-zero DetA is consistent with predictions failing spatial or semantic matching gates.",
            "AssA": "Very low AssA is expected when few detections match; identity quality cannot be isolated yet.",
            "LocA": "LocA of 8-18 percent indicates poor localization even among the limited accepted matches.",
        },
        "recommended_v6_fix": _v6_recommendation(verdict, best, causes),
        "do_not_optimize_before_fix": [
            "ReID embeddings", "MTMC association thresholds", "coverage extension",
            "track fragmentation cleanup", "additional V3/V4/V5 geometry tuning on the current convention",
        ],
        "upload_recommendation": _upload_recommendation(verdict),
    }
    write_json(directory / "likely_failure_causes.json", likely)
    return likely


def _category_diagnosis(
    category: str, original: Dict[str, Any], individual: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    candidates = [row for row in individual if row.get("category") == category]
    best = sorted(candidates, key=lambda row: _category_rank_key(category, row))[0] if candidates else {}
    original_rate = _number(original.get("match_rate_at_2m"), 0.0)
    best_rate = _number(best.get("match_rate_at_2m"), 0.0)
    original_median = _number(original.get("center_error_median"), float("inf"))
    best_median = _number(best.get("center_error_median"), float("inf"))
    operation = _operation(best)
    is_original = operation in {
        "original", "center_original", "w_l_h_original", "yaw_original",
        "frame_original", "official_mapping",
    }
    gain = best_rate - original_rate
    ratio = best_median / original_median if original_median not in (0.0, float("inf")) else None
    meaningful = not is_original and (gain >= 0.05 or (ratio is not None and ratio <= 0.75))
    diagnostic_metric = "spatial_match_rate_and_center_error"
    metric_before: Any = original_rate
    metric_after: Any = best_rate
    if category == "dimension":
        diagnostic_metric = "dimension_ratio_log_deviation_and_iou"
        metric_before = _dimension_deviation(original)
        metric_after = _dimension_deviation(best)
        original_iou = _number(original.get("iou3d_proxy_median"), 0.0)
        best_iou = _number(best.get("iou3d_proxy_median"), 0.0)
        meaningful = not is_original and (
            (metric_before != float("inf") and metric_after <= metric_before * 0.75)
            or best_iou - original_iou >= 0.05
        )
    elif category == "yaw":
        diagnostic_metric = "yaw_error_median"
        metric_before = _number(original.get("yaw_error_median"), float("inf"))
        metric_after = _number(best.get("yaw_error_median"), float("inf"))
        meaningful = not is_original and metric_after < metric_before and (
            metric_after <= metric_before * 0.75 or metric_before - metric_after >= 0.2
        )
    return {
        "category": category, "best_hypothesis": best, "best_operation": operation,
        "original_match_rate_at_2m": original_rate, "best_match_rate_at_2m": best_rate,
        "absolute_match_rate_gain": gain, "original_center_error_median": original_median,
        "best_center_error_median": best_median, "center_error_median_ratio": ratio,
        "diagnostic_metric": diagnostic_metric, "diagnostic_metric_before": metric_before,
        "diagnostic_metric_after": metric_after,
        "meaningful_improvement": meaningful,
        "diagnosis": "%s convention mismatch is plausible" % category if meaningful else "no clear %s convention fix" % category,
    }


def _verdict(
    diagnoses: Dict[str, Dict[str, Any]], source_summary: Dict[str, Any],
) -> Tuple[str, List[str]]:
    if source_summary.get("status") != "ok":
        return "val_prediction_source_missing_fix_required", ["comparable validation predictions were not found"]
    meaningful = [category for category, item in diagnoses.items() if item.get("meaningful_improvement")]
    if len(meaningful) > 1:
        return "multiple_convention_mismatches_likely", meaningful
    if not meaningful:
        return "no_clear_convention_fix_found", ["no individual convention hypothesis materially improved local matching"]
    category = meaningful[0]
    operation = str(diagnoses[category].get("best_operation", ""))
    if category == "axis":
        if operation.startswith("scale_xyz"):
            return "unit_scale_mismatch_likely", [operation]
        if operation.startswith("swap") or operation.startswith("flip"):
            return "axis_permutation_or_sign_error_likely", [operation]
        return "coordinate_frame_mismatch_likely", [operation]
    mapping = {
        "center": "center_convention_mismatch_likely", "dimension": "dimension_order_mismatch_likely",
        "yaw": "yaw_convention_mismatch_likely", "frame": "frame_offset_mismatch_likely",
        "class": "class_mapping_mismatch_likely",
    }
    return mapping.get(category, "no_clear_convention_fix_found"), [operation]


def _v6_recommendation(verdict: str, best: Dict[str, Any], causes: Sequence[str]) -> str:
    operations = best.get("operations", {}) if isinstance(best, dict) else {}
    if verdict == "val_prediction_source_missing_fix_required":
        return "Create a GT-independent validation export from the same pre-submission pipeline before changing V6 geometry."
    if verdict == "no_clear_convention_fix_found":
        return "Treat pseudo-3D localization quality or camera-to-world projection as the primary V6 investigation; do not apply a blind convention transform."
    return "Implement the validated transform at the shared Track1 geometry export boundary, then regenerate validation diagnostics before upload. Candidate operations: %s. Causes: %s" % (operations, list(causes))


def _upload_recommendation(verdict: str) -> str:
    if verdict == "no_clear_convention_fix_found":
        return "Do not upload another existing variant; diagnose projection/localization quality first."
    if verdict == "val_prediction_source_missing_fix_required":
        return "Do not upload; validation evidence is missing."
    return "Do not upload another existing frozen variant. Build V6 only after applying and validating the identified convention fix."


def _operation(row: Dict[str, Any]) -> str:
    operations = row.get("operations", {})
    if isinstance(operations, dict) and operations:
        return str(next(iter(operations.values())))
    return ""


def _rank_key(row: Dict[str, Any]) -> Tuple[float, float, str]:
    return (
        -_number(row.get("match_rate_at_2m"), -1.0),
        _number(row.get("center_error_median"), float("inf")),
        str(row.get("hypothesis", "")),
    )


def _category_rank_key(category: str, row: Dict[str, Any]) -> Tuple[Any, ...]:
    if category == "dimension":
        return (_dimension_deviation(row), -_number(row.get("iou3d_proxy_median"), -1.0), str(row.get("hypothesis", "")))
    if category == "yaw":
        return (_number(row.get("yaw_error_median"), float("inf")), str(row.get("hypothesis", "")))
    return _rank_key(row)


def _dimension_deviation(row: Dict[str, Any]) -> float:
    import math

    values = [
        _number(row.get("dimension_ratio_width_median"), float("inf")),
        _number(row.get("dimension_ratio_length_median"), float("inf")),
        _number(row.get("dimension_ratio_height_median"), float("inf")),
    ]
    if any(value <= 0.0 or value == float("inf") for value in values):
        return float("inf")
    return sum(abs(math.log(value)) for value in values)


def _number(value: Any, default: float) -> float:
    try:
        return float(value) if value is not None else default
    except (TypeError, ValueError):
        return default
