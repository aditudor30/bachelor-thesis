"""Projection checks for pseudo-3D predictions."""

from typing import Any, Dict, List

import numpy as np

from deep_oc_sort_3d.visualization3d.cuboid_projection import project_cuboid_to_image


def check_prediction_projection(prediction: Dict[str, Any], calibration: Any) -> Dict[str, Any]:
    """Check whether one pseudo-3D cuboid projects through calibration."""
    center = _array3([prediction.get("center_x"), prediction.get("center_y"), prediction.get("center_z")])
    dims = _array3([prediction.get("width_3d"), prediction.get("length_3d"), prediction.get("height_3d")])
    yaw = prediction.get("yaw")
    if center is None:
        return {"projection_valid": False, "projection_error_reason": "missing_center_3d"}
    if dims is None:
        return {"projection_valid": False, "projection_error_reason": "invalid_dimensions"}
    try:
        result = project_cuboid_to_image(center, dims, float(yaw), calibration)
    except (TypeError, ValueError):
        return {"projection_valid": False, "projection_error_reason": "invalid_yaw"}
    if not result.get("success"):
        return {"projection_valid": False, "projection_error_reason": result.get("error_message", "projection_error")}
    return {"projection_valid": True, "projection_error_reason": ""}


def summarize_projection_checks(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Summarize projection check rows."""
    total = len(rows)
    success = sum(1 for row in rows if row.get("projection_valid"))
    reasons = {}
    for row in rows:
        if row.get("projection_valid"):
            continue
        reason = str(row.get("projection_error_reason", "unknown"))
        reasons[reason] = reasons.get(reason, 0) + 1
    return {
        "total": total,
        "projection_success": success,
        "projection_failed": total - success,
        "projection_success_rate": float(success) / float(total) if total else None,
        "failure_reasons": reasons,
    }


def _array3(values: Any) -> Any:
    try:
        arr = np.asarray(values, dtype=float).reshape(-1)
    except (TypeError, ValueError):
        return None
    if arr.size < 3 or not np.all(np.isfinite(arr[:3])):
        return None
    return arr[:3]

