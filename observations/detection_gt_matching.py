"""Match YOLO detections to visible ground-truth boxes."""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from deep_oc_sort_3d.data.ground_truth import GroundTruthObject
from deep_oc_sort_3d.detection2d.yolo_types import Detection2D


DEFAULT_CLASS_MAPPING = {
    "Person": 0,
    "Forklift": 1,
    "PalletTruck": 2,
    "Transporter": 3,
    "FourierGR1T2": 4,
    "AgilityDigit": 5,
    "NovaCarter": 6,
}


def bbox_iou_xyxy(
    box_a: Tuple[float, float, float, float],
    box_b: Tuple[float, float, float, float],
) -> float:
    """Compute IoU between two xyxy boxes."""
    ax1, ay1, ax2, ay2 = _ordered_box(box_a)
    bx1, by1, bx2, by2 = _ordered_box(box_b)
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter_area
    if union <= 0.0:
        return 0.0
    return float(inter_area / union)


def match_detections_to_gt(
    detections: List[Detection2D],
    gt_objects: List[GroundTruthObject],
    camera_id: str,
    iou_threshold: float = 0.3,
    class_must_match: bool = True,
    use_hungarian: bool = True,
) -> Tuple[Dict[int, GroundTruthObject], Dict[int, float]]:
    """Match detections to visible GT objects with Hungarian or greedy fallback."""
    gt_visible = _visible_gt_objects(gt_objects, camera_id)
    if not detections or not gt_visible:
        return ({}, {})
    iou_matrix = _build_iou_matrix(detections, gt_visible, camera_id, class_must_match)
    if use_hungarian:
        try:
            matches = _hungarian_matches(iou_matrix, iou_threshold)
        except Exception:
            matches = _greedy_matches(iou_matrix, iou_threshold)
    else:
        matches = _greedy_matches(iou_matrix, iou_threshold)

    matched_gt = {}
    matched_iou = {}
    for det_idx, gt_idx, iou in matches:
        matched_gt[det_idx] = gt_visible[gt_idx]
        matched_iou[det_idx] = float(iou)
    return (matched_gt, matched_iou)


def compute_matching_stats(
    detections: List[Detection2D],
    gt_objects: List[GroundTruthObject],
    camera_id: str,
    iou_threshold: float = 0.3,
    class_must_match: bool = True,
) -> Dict[str, Any]:
    """Compute matching precision/recall and per-class counts."""
    gt_visible = _visible_gt_objects(gt_objects, camera_id)
    matches, ious = match_detections_to_gt(
        detections,
        gt_objects,
        camera_id,
        iou_threshold=iou_threshold,
        class_must_match=class_must_match,
    )
    num_detections = len(detections)
    num_gt_visible = len(gt_visible)
    num_matches = len(matches)
    precision = 0.0 if num_detections == 0 else float(num_matches) / float(num_detections)
    recall = 0.0 if num_gt_visible == 0 else float(num_matches) / float(num_gt_visible)
    mean_iou = None if not ious else float(np.mean(list(ious.values())))
    return {
        "num_detections": num_detections,
        "num_gt_visible": num_gt_visible,
        "num_matches": num_matches,
        "precision_at_iou": precision,
        "recall_at_iou": recall,
        "mean_iou": mean_iou,
        "per_class": _per_class_stats(detections, gt_visible, matches),
    }


def _build_iou_matrix(
    detections: List[Detection2D],
    gt_visible: List[GroundTruthObject],
    camera_id: str,
    class_must_match: bool,
) -> np.ndarray:
    matrix = np.zeros((len(detections), len(gt_visible)), dtype=float)
    for det_idx, det in enumerate(detections):
        for gt_idx, gt in enumerate(gt_visible):
            if class_must_match and det.class_id != _class_id_for_name(gt.object_type):
                continue
            matrix[det_idx, gt_idx] = bbox_iou_xyxy(det.bbox_xyxy, gt.visible_bboxes_2d[camera_id])
    return matrix


def _hungarian_matches(iou_matrix: np.ndarray, iou_threshold: float) -> List[Tuple[int, int, float]]:
    from scipy.optimize import linear_sum_assignment

    if iou_matrix.size == 0:
        return []
    cost = 1.0 - iou_matrix
    det_indices, gt_indices = linear_sum_assignment(cost)
    matches = []
    for det_idx, gt_idx in zip(det_indices, gt_indices):
        iou = float(iou_matrix[det_idx, gt_idx])
        if iou >= iou_threshold:
            matches.append((int(det_idx), int(gt_idx), iou))
    return matches


def _greedy_matches(iou_matrix: np.ndarray, iou_threshold: float) -> List[Tuple[int, int, float]]:
    pairs = []
    for det_idx in range(iou_matrix.shape[0]):
        for gt_idx in range(iou_matrix.shape[1]):
            iou = float(iou_matrix[det_idx, gt_idx])
            if iou >= iou_threshold:
                pairs.append((det_idx, gt_idx, iou))
    pairs = sorted(pairs, key=lambda item: item[2], reverse=True)
    used_dets = set()
    used_gts = set()
    matches = []
    for det_idx, gt_idx, iou in pairs:
        if det_idx in used_dets or gt_idx in used_gts:
            continue
        used_dets.add(det_idx)
        used_gts.add(gt_idx)
        matches.append((det_idx, gt_idx, iou))
    return matches


def _visible_gt_objects(gt_objects: List[GroundTruthObject], camera_id: str) -> List[GroundTruthObject]:
    return [obj for obj in gt_objects if camera_id in obj.visible_bboxes_2d]


def _class_id_for_name(class_name: str) -> int:
    if class_name in DEFAULT_CLASS_MAPPING:
        return int(DEFAULT_CLASS_MAPPING[class_name])
    lower = {}
    for name, class_id in DEFAULT_CLASS_MAPPING.items():
        lower[name.lower()] = int(class_id)
    return int(lower.get(str(class_name).lower(), -1))


def _ordered_box(box: Tuple[float, float, float, float]) -> Tuple[float, float, float, float]:
    x1, y1, x2, y2 = box
    return (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))


def _per_class_stats(
    detections: List[Detection2D],
    gt_visible: List[GroundTruthObject],
    matches: Dict[int, GroundTruthObject],
) -> Dict[str, Dict[str, int]]:
    stats = {}
    for det in detections:
        key = det.class_name
        if key not in stats:
            stats[key] = {"detections": 0, "gt_visible": 0, "matches": 0}
        stats[key]["detections"] += 1
    for gt in gt_visible:
        key = gt.object_type
        if key not in stats:
            stats[key] = {"detections": 0, "gt_visible": 0, "matches": 0}
        stats[key]["gt_visible"] += 1
    for _det_idx, gt in matches.items():
        key = gt.object_type
        if key not in stats:
            stats[key] = {"detections": 0, "gt_visible": 0, "matches": 0}
        stats[key]["matches"] += 1
    return stats

