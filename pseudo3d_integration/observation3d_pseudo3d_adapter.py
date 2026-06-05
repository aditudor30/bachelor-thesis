"""Merge stabilized pseudo-3D predictions into Observation3D dictionaries."""

from typing import Any, Dict, Optional

import numpy as np

from deep_oc_sort_3d.observations.observation_io import observation_to_dict
from deep_oc_sort_3d.pseudo3d.pseudo3d_types import Pseudo3DOutput


def merge_observation_with_pseudo3d(
    observation: Any,
    pseudo3d_prediction: Optional[Pseudo3DOutput],
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """Return an Observation3D-compatible dict enriched with pseudo-3D fields."""
    row = _observation_dict(observation)
    policy = _policy(config)
    pseudo_valid = _prediction_has_center(pseudo3d_prediction)
    if bool(policy.get("use_stabilized_pseudo3d", True)) and pseudo_valid:
        _apply_prediction(row, pseudo3d_prediction)
        row.update(build_pseudo3d_metadata_dict(pseudo3d_prediction, None))
        row["pseudo3d_matched"] = True
        row["pseudo3d_used"] = True
        row["fallback_original_used"] = False
        row["has_3d"] = True
    else:
        fallback_reason = "pseudo3d_missing" if pseudo3d_prediction is None else "pseudo3d_invalid"
        row.update(build_pseudo3d_metadata_dict(pseudo3d_prediction, fallback_reason))
        _apply_fallback(row, pseudo3d_prediction, config, fallback_reason)
    _ensure_dimensions(row, pseudo3d_prediction, config)
    _ensure_yaw(row, pseudo3d_prediction)
    row["source"] = str(policy.get("output_source", "baseline_v2_pseudo3d"))
    return row


def build_pseudo3d_metadata_dict(
    prediction: Optional[Pseudo3DOutput],
    fallback_reason: Optional[str] = None,
) -> Dict[str, Any]:
    """Build provenance metadata for one integrated observation."""
    if prediction is None:
        return {
            "pseudo3d_confidence": None,
            "coordinate_frame": "unknown",
            "center_3d_source": "missing_pseudo3d",
            "dimensions_3d_source": "unknown",
            "yaw_source": "class_default",
            "depth_source": "unknown",
            "is_gt_derived": False,
            "is_estimated_for_test": True,
            "pseudo3d_method": "missing",
            "pseudo3d_version": "",
            "projection_valid": None,
            "projection_error_reason": None,
            "fallback_reason": fallback_reason,
        }
    return {
        "pseudo3d_confidence": float(prediction.confidence_3d),
        "coordinate_frame": prediction.coordinate_frame,
        "center_3d_source": prediction.center_3d_source,
        "dimensions_3d_source": prediction.dimensions_3d_source,
        "yaw_source": prediction.yaw_source,
        "depth_source": prediction.depth_source,
        "is_gt_derived": False,
        "is_estimated_for_test": True,
        "pseudo3d_method": prediction.pseudo3d_method,
        "pseudo3d_version": prediction.pseudo3d_version,
        "projection_valid": prediction.projection_valid,
        "projection_error_reason": prediction.projection_error_reason,
        "fallback_reason": fallback_reason,
    }


def _apply_prediction(row: Dict[str, Any], prediction: Pseudo3DOutput) -> None:
    row["center_3d"] = _array_to_list(prediction.center_3d)
    row["dimensions_3d"] = _array_to_list(prediction.dimensions_3d)
    row["yaw"] = None if prediction.yaw is None else float(prediction.yaw)
    row["depth_value"] = None if prediction.depth is None else float(prediction.depth)
    row["depth_sampling_method"] = "pseudo3d_stabilized"
    row["source_notes"] = prediction.source_notes


def _apply_fallback(row: Dict[str, Any], prediction: Optional[Pseudo3DOutput], config: Dict[str, Any], fallback_reason: str) -> None:
    policy = _policy(config)
    original_center_valid = _list3(row.get("center_3d")) is not None
    if bool(policy.get("fallback_to_original_3d_if_missing", True)) and original_center_valid:
        row["center_3d_source"] = "original_fallback"
        if _list3(row.get("dimensions_3d")) is not None:
            row["dimensions_3d_source"] = "original_fallback"
        if row.get("yaw") not in (None, ""):
            row["yaw_source"] = "original_fallback"
        if row.get("depth_value") not in (None, ""):
            row["depth_source"] = "original_fallback"
        row["fallback_original_used"] = True
        row["has_3d"] = True
        row["source_notes"] = _append_note(str(row.get("source_notes") or ""), "%s_fallback_original" % fallback_reason)
    elif bool(policy.get("drop_if_no_center_3d", False)):
        row["center_3d"] = None
        row["fallback_original_used"] = False
        row["has_3d"] = False
        row["source_notes"] = _append_note(str(row.get("source_notes") or ""), "%s_no_center" % fallback_reason)
    else:
        row["fallback_original_used"] = False
        row["has_3d"] = original_center_valid
        row["source_notes"] = _append_note(str(row.get("source_notes") or ""), "%s_marked_missing" % fallback_reason)
    row["pseudo3d_matched"] = prediction is not None
    row["pseudo3d_used"] = False
    row["fallback_reason"] = fallback_reason


def _ensure_dimensions(row: Dict[str, Any], prediction: Optional[Pseudo3DOutput], config: Dict[str, Any]) -> None:
    if _list3(row.get("dimensions_3d")) is not None:
        return
    policy = _policy(config)
    if prediction is not None and prediction.dimensions_3d is not None:
        row["dimensions_3d"] = _array_to_list(prediction.dimensions_3d)
        row["dimensions_3d_source"] = prediction.dimensions_3d_source
        return
    if not bool(policy.get("fallback_to_class_prior_dimensions", True)):
        return
    priors = config.get("class_priors_by_id", {})
    prior = priors.get(str(row.get("class_id")))
    if prior is None:
        prior = priors.get(int(row.get("class_id", -1))) if row.get("class_id") not in (None, "") else None
    if prior is not None:
        row["dimensions_3d"] = [float(prior["width"]), float(prior["length"]), float(prior["height"])]
        row["dimensions_3d_source"] = "class_prior"
        row["class_prior_dimensions_used"] = True


def _ensure_yaw(row: Dict[str, Any], prediction: Optional[Pseudo3DOutput]) -> None:
    if row.get("yaw") not in (None, ""):
        return
    if prediction is not None and prediction.yaw is not None:
        row["yaw"] = float(prediction.yaw)
        row["yaw_source"] = prediction.yaw_source
    else:
        row["yaw"] = 0.0
        row["yaw_source"] = "class_default"


def _policy(config: Dict[str, Any]) -> Dict[str, Any]:
    section = config.get("pseudo3d_integration", config)
    return section if isinstance(section, dict) else {}


def _observation_dict(observation: Any) -> Dict[str, Any]:
    if isinstance(observation, dict):
        return dict(observation)
    return observation_to_dict(observation)


def _prediction_has_center(prediction: Optional[Pseudo3DOutput]) -> bool:
    if prediction is None or prediction.center_3d is None:
        return False
    return _array3(prediction.center_3d) is not None


def _array_to_list(value: Any) -> Any:
    array = _array3(value)
    if array is None:
        return None
    return [float(item) for item in array]


def _array3(value: Any) -> Any:
    try:
        array = np.asarray(value, dtype=float).reshape(-1)
    except (TypeError, ValueError):
        return None
    if array.size < 3 or not np.all(np.isfinite(array[:3])):
        return None
    return array[:3]


def _list3(value: Any) -> Any:
    return _array3(value)


def _append_note(existing: str, note: str) -> str:
    if note in existing:
        return existing
    if existing:
        return "%s; %s" % (existing, note)
    return note
