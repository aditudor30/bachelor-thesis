"""Evaluation helpers for pseudo-3D predictions."""

import math
from typing import Any, Dict, List, Optional

import numpy as np

from deep_oc_sort_3d.audit3d.audit3d_io import numeric_stats, optional_float
from deep_oc_sort_3d.pseudo3d.pseudo3d_types import Pseudo3DEvalRecord, pseudo3d_eval_record_to_dict


def evaluate_prediction_dicts(
    predictions: List[Dict[str, Any]],
    gt_lookup: Optional[Dict[Any, Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Evaluate predictions against an optional GT lookup."""
    gt_lookup = gt_lookup or {}
    records = []
    for prediction in predictions:
        key = _prediction_key(prediction)
        gt = gt_lookup.get(key)
        records.append(build_eval_record(prediction, gt))
    return summarize_eval_records(records)


def build_eval_record(prediction: Dict[str, Any], gt: Optional[Dict[str, Any]]) -> Pseudo3DEvalRecord:
    """Build one evaluation record."""
    pred_center = _center_from_prediction(prediction)
    gt_center = _center_from_gt(gt) if gt is not None else None
    pred_dims = _dims_from_prediction(prediction)
    gt_dims = _dims_from_gt(gt) if gt is not None else None
    center_error = _distance(pred_center, gt_center)
    return Pseudo3DEvalRecord(
        subset=str(prediction.get("subset", "")),
        scene_name=str(prediction.get("scene_name", "")),
        camera_id=str(prediction.get("camera_id", "")),
        frame_id=int(float(prediction.get("frame_id", 0))),
        class_id=int(float(prediction.get("class_id", -1))),
        class_name=str(prediction.get("class_name", "")),
        local_track_id=_optional_int(prediction.get("local_track_id")),
        global_track_id=_optional_int(prediction.get("global_track_id")),
        predicted_center_3d=pred_center,
        gt_center_3d=gt_center,
        center_error=center_error,
        depth_error=_abs_delta(_z(pred_center), _z(gt_center)),
        dimensions_error=_distance(pred_dims, gt_dims),
        yaw_error=_yaw_error(optional_float(prediction.get("yaw")), optional_float(gt.get("yaw") if gt else None)),
        projection_valid=_optional_bool(prediction.get("projection_valid")),
        evaluation_available=gt is not None,
        missing_gt_reason=None if gt is not None else "missing_gt_match",
    )


def summarize_eval_records(records: List[Pseudo3DEvalRecord]) -> Dict[str, Any]:
    """Summarize pseudo-3D evaluation records."""
    rows = [pseudo3d_eval_record_to_dict(record) for record in records]
    available = [record for record in records if record.evaluation_available]
    return {
        "num_predictions": len(records),
        "num_evaluated": len(available),
        "num_missing_gt": len(records) - len(available),
        "center_error": numeric_stats([record.center_error for record in available]),
        "depth_error": numeric_stats([record.depth_error for record in available]),
        "dimension_error": numeric_stats([record.dimensions_error for record in available]),
        "yaw_error": numeric_stats([record.yaw_error for record in available]),
        "projection_success_rate": _projection_success_rate(records),
        "records": rows,
    }


def _prediction_key(prediction: Dict[str, Any]) -> Any:
    return (
        prediction.get("scene_name"),
        prediction.get("camera_id"),
        int(float(prediction.get("frame_id", 0))),
        int(float(prediction.get("class_id", -1))),
        prediction.get("local_track_id"),
    )


def _center_from_prediction(row: Dict[str, Any]) -> Optional[np.ndarray]:
    if isinstance(row.get("center_3d"), list):
        return _array3(row.get("center_3d"))
    return _array3([row.get("center_x"), row.get("center_y"), row.get("center_z")])


def _dims_from_prediction(row: Dict[str, Any]) -> Optional[np.ndarray]:
    if isinstance(row.get("dimensions_3d"), list):
        return _array3(row.get("dimensions_3d"))
    return _array3([row.get("width_3d"), row.get("length_3d"), row.get("height_3d")])


def _center_from_gt(row: Optional[Dict[str, Any]]) -> Optional[np.ndarray]:
    if row is None:
        return None
    return _array3(row.get("center_3d", [row.get("center_x"), row.get("center_y"), row.get("center_z")]))


def _dims_from_gt(row: Optional[Dict[str, Any]]) -> Optional[np.ndarray]:
    if row is None:
        return None
    return _array3(row.get("dimensions_3d", [row.get("width_3d"), row.get("length_3d"), row.get("height_3d")]))


def _array3(values: Any) -> Optional[np.ndarray]:
    try:
        arr = np.asarray(values, dtype=float).reshape(-1)
    except (TypeError, ValueError):
        return None
    if arr.size < 3 or not np.all(np.isfinite(arr[:3])):
        return None
    return arr[:3]


def _distance(a: Optional[np.ndarray], b: Optional[np.ndarray]) -> Optional[float]:
    if a is None or b is None:
        return None
    return float(np.linalg.norm(a - b))


def _z(value: Optional[np.ndarray]) -> Optional[float]:
    if value is None:
        return None
    return float(value[2])


def _abs_delta(a: Optional[float], b: Optional[float]) -> Optional[float]:
    if a is None or b is None:
        return None
    return abs(float(a) - float(b))


def _yaw_error(a: Optional[float], b: Optional[float]) -> Optional[float]:
    if a is None or b is None:
        return None
    delta = float(a) - float(b)
    while delta > math.pi:
        delta -= 2.0 * math.pi
    while delta < -math.pi:
        delta += 2.0 * math.pi
    return abs(delta)


def _projection_success_rate(records: List[Pseudo3DEvalRecord]) -> Optional[float]:
    values = [record.projection_valid for record in records if record.projection_valid is not None]
    if not values:
        return None
    if not any(values):
        # Raw prediction files usually leave projection_valid unset for successes
        # and only carry False on extraction failures. Projection quality is
        # measured by check_pseudo3d_projection.py, so avoid a misleading 0.0.
        return None
    return float(sum(1 for value in values if value)) / float(len(values))


def _optional_int(value: Any) -> Optional[int]:
    if value in (None, ""):
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _optional_bool(value: Any) -> Optional[bool]:
    if value in (None, ""):
        return None
    if isinstance(value, bool):
        return value
    return str(value).lower() in ("true", "1", "yes")
