"""Conservative pseudo3D prediction to train/val GT matching."""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from deep_oc_sort_3d.data.ground_truth import GroundTruthObject
from deep_oc_sort_3d.observations.detection_gt_matching import bbox_iou_xyxy
from deep_oc_sort_3d.v5_geometry_calibration.geometry_calibration_io import vector3


def match_prediction_to_gt(
    prediction: Dict[str, Any],
    candidates: List[GroundTruthObject],
    camera_id: str,
    config: Dict[str, Any],
) -> Tuple[Optional[GroundTruthObject], Optional[float], str]:
    """Match by trusted object ID, then IoU, then center distance."""
    rules = config.get("matching", {})
    matched_object_id = prediction.get("matched_gt_object_id")
    object_id = _optional_int(matched_object_id if matched_object_id not in (None, "") else prediction.get("object_id"))
    if _truthy(prediction.get("matched_gt")) and object_id is not None:
        direct = [item for item in candidates if int(item.object_id) == object_id]
        if len(direct) == 1:
            return direct[0], _bbox_iou(prediction, direct[0], camera_id), "trusted_object_id"

    bbox = _bbox(prediction)
    if bbox is not None and bool(rules.get("max_2d_iou_match_if_available", True)):
        scored = []
        for candidate in candidates:
            gt_bbox = candidate.visible_bboxes_2d.get(camera_id)
            if gt_bbox is not None:
                scored.append((bbox_iou_xyxy(bbox, gt_bbox), candidate))
        scored.sort(key=lambda item: item[0], reverse=True)
        minimum = float(rules.get("min_2d_iou", 0.2))
        if scored and scored[0][0] >= minimum:
            if _ambiguous_scores(scored, float(rules.get("ambiguity_iou_margin", 0.05))):
                return None, None, "ambiguous_iou"
            return scored[0][1], float(scored[0][0]), "bbox_iou"

    center = vector3(prediction.get("center_3d"))
    if center is not None:
        scored_distance = sorted(
            [(float(np.linalg.norm(center - item.location_3d)), item) for item in candidates],
            key=lambda item: item[0],
        )
        maximum = float(rules.get("max_center_distance_m", 5.0))
        if scored_distance and scored_distance[0][0] <= maximum:
            margin = float(rules.get("ambiguity_distance_margin_m", 0.25))
            if len(scored_distance) > 1 and scored_distance[1][0] - scored_distance[0][0] < margin:
                return None, None, "ambiguous_center_distance"
            return scored_distance[0][1], None, "center_distance"
    return None, None, "no_conservative_match"


def _ambiguous_scores(values: List[Tuple[float, GroundTruthObject]], margin: float) -> bool:
    return len(values) > 1 and values[0][0] - values[1][0] < margin


def _bbox(prediction: Dict[str, Any]) -> Optional[Tuple[float, float, float, float]]:
    value = prediction.get("bbox_xyxy")
    if isinstance(value, (list, tuple)) and len(value) >= 4:
        try:
            return (float(value[0]), float(value[1]), float(value[2]), float(value[3]))
        except (TypeError, ValueError):
            return None
    try:
        return (float(prediction["x1"]), float(prediction["y1"]), float(prediction["x2"]), float(prediction["y2"]))
    except (KeyError, TypeError, ValueError):
        return None


def _bbox_iou(prediction: Dict[str, Any], gt: GroundTruthObject, camera_id: str) -> Optional[float]:
    bbox = _bbox(prediction)
    gt_bbox = gt.visible_bboxes_2d.get(camera_id)
    return None if bbox is None or gt_bbox is None else bbox_iou_xyxy(bbox, gt_bbox)


def _optional_int(value: Any) -> Optional[int]:
    try:
        if value in (None, ""):
            return None
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _truthy(value: Any) -> bool:
    return value is True or str(value).lower() in ("1", "true", "yes")
