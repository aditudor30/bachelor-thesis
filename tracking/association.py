"""Association costs and matching for local single-camera tracking."""

from typing import Any, Dict, List, Tuple

import numpy as np

from deep_oc_sort_3d.tracking.motion_model import bbox_center_xyxy
from deep_oc_sort_3d.tracking.track_state import LocalTrack
from deep_oc_sort_3d.tracking.track_types import LocalTrackDetection


INF_COST = 1e9


def bbox_iou_xyxy(
    box_a: Tuple[float, float, float, float],
    box_b: Tuple[float, float, float, float],
) -> float:
    """Compute IoU between two xyxy bboxes."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_x1 = max(min(ax1, ax2), min(bx1, bx2))
    inter_y1 = max(min(ay1, ay2), min(by1, by2))
    inter_x2 = min(max(ax1, ax2), max(bx1, bx2))
    inter_y2 = min(max(ay1, ay2), max(by1, by2))
    inter_w = max(0.0, float(inter_x2) - float(inter_x1))
    inter_h = max(0.0, float(inter_y2) - float(inter_y1))
    inter_area = inter_w * inter_h
    area_a = max(0.0, abs(float(ax2) - float(ax1))) * max(0.0, abs(float(ay2) - float(ay1)))
    area_b = max(0.0, abs(float(bx2) - float(bx1))) * max(0.0, abs(float(by2) - float(by1)))
    denom = area_a + area_b - inter_area
    if denom <= 0.0:
        return 0.0
    return float(inter_area) / float(denom)


def center_distance_3d(center_a: object, center_b: object) -> object:
    """Return Euclidean 3D center distance, or None when either center is missing."""
    if center_a is None or center_b is None:
        return None
    a = np.asarray(center_a, dtype=float).reshape(-1)
    b = np.asarray(center_b, dtype=float).reshape(-1)
    if a.size < 3 or b.size < 3:
        return None
    return float(np.linalg.norm(a[:3] - b[:3]))


def center_distance_2d_from_bbox(
    box_a: Tuple[float, float, float, float],
    box_b: Tuple[float, float, float, float],
) -> float:
    """Return Euclidean distance between bbox centers."""
    return float(np.linalg.norm(bbox_center_xyxy(box_a) - bbox_center_xyxy(box_b)))


def compute_association_cost(
    track: LocalTrack,
    det: LocalTrackDetection,
    frame_id: int,
    mode: str = "hybrid",
    max_3d_distance: float = 3.0,
    max_2d_center_distance: float = 200.0,
    iou_weight: float = 0.5,
    distance_3d_weight: float = 0.4,
    distance_2d_weight: float = 0.3,
    confidence_weight: float = 0.1,
    class_must_match: bool = True,
    min_iou: float = 0.0,
) -> float:
    """Compute association cost between one track and one detection."""
    if class_must_match and int(track.class_id) != int(det.class_id):
        return INF_COST

    predicted = track.predict(frame_id)
    last_det = track.last_detection
    use_3d = mode == "oracle_3d" or (
        mode == "hybrid" and predicted.get("center_3d") is not None and det.center_3d is not None
    )
    if use_3d:
        distance = center_distance_3d(predicted.get("center_3d"), det.center_3d)
        if distance is None:
            return INF_COST
        if distance > float(max_3d_distance):
            return INF_COST
        cost = float(distance_3d_weight) * (float(distance) / max(float(max_3d_distance), 1e-6))
        cost -= float(confidence_weight) * float(det.confidence)
        return _clamp_cost(cost)

    if mode == "oracle_3d":
        return INF_COST

    iou = bbox_iou_xyxy(last_det.bbox_xyxy, det.bbox_xyxy)
    if iou < float(min_iou):
        return INF_COST
    distance_2d = center_distance_2d_from_bbox(last_det.bbox_xyxy, det.bbox_xyxy)
    norm_distance = min(distance_2d / max(float(max_2d_center_distance), 1e-6), 1.0)
    cost = float(iou_weight) * (1.0 - float(iou)) + float(distance_2d_weight) * norm_distance
    cost -= float(confidence_weight) * float(det.confidence)
    return _clamp_cost(cost)


def associate_detections_to_tracks(
    detections: List[LocalTrackDetection],
    tracks: List[LocalTrack],
    frame_id: int,
    config: Dict[str, Any],
) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
    """Associate detections to tracks using Hungarian matching when available."""
    if not tracks:
        return [], [], list(range(len(detections)))
    if not detections:
        return [], list(range(len(tracks))), []

    cost_matrix = np.zeros((len(tracks), len(detections)), dtype=float)
    for track_index, track in enumerate(tracks):
        for det_index, det in enumerate(detections):
            cost_matrix[track_index, det_index] = compute_association_cost(
                track=track,
                det=det,
                frame_id=frame_id,
                mode=str(config.get("mode", "hybrid")),
                max_3d_distance=float(config.get("max_3d_distance", 3.0)),
                max_2d_center_distance=float(config.get("max_2d_center_distance", 200.0)),
                iou_weight=float(config.get("iou_weight", 0.5)),
                distance_3d_weight=float(config.get("distance_3d_weight", 0.4)),
                distance_2d_weight=float(config.get("distance_2d_weight", 0.3)),
                confidence_weight=float(config.get("confidence_weight", 0.1)),
                class_must_match=bool(config.get("class_must_match", True)),
                min_iou=float(config.get("min_iou", 0.0)),
            )
    matched = _match_cost_matrix(cost_matrix, float(config.get("cost_threshold", 0.7)))
    matched_tracks = set(track_idx for track_idx, _det_idx in matched)
    matched_detections = set(det_idx for _track_idx, det_idx in matched)
    unmatched_tracks = [idx for idx in range(len(tracks)) if idx not in matched_tracks]
    unmatched_detections = [idx for idx in range(len(detections)) if idx not in matched_detections]
    return matched, unmatched_tracks, unmatched_detections


def _match_cost_matrix(cost_matrix: np.ndarray, cost_threshold: float) -> List[Tuple[int, int]]:
    try:
        from scipy.optimize import linear_sum_assignment
    except ImportError:
        return _greedy_match(cost_matrix, cost_threshold)
    row_indices, col_indices = linear_sum_assignment(cost_matrix)
    matched = []
    for row, col in zip(row_indices, col_indices):
        cost = float(cost_matrix[row, col])
        if cost <= float(cost_threshold) and cost < INF_COST:
            matched.append((int(row), int(col)))
    return matched


def _greedy_match(cost_matrix: np.ndarray, cost_threshold: float) -> List[Tuple[int, int]]:
    pairs = []
    for row in range(cost_matrix.shape[0]):
        for col in range(cost_matrix.shape[1]):
            pairs.append((float(cost_matrix[row, col]), int(row), int(col)))
    pairs = sorted(pairs, key=lambda item: item[0])
    used_rows = set()
    used_cols = set()
    matched = []
    for cost, row, col in pairs:
        if cost > float(cost_threshold) or cost >= INF_COST:
            continue
        if row in used_rows or col in used_cols:
            continue
        matched.append((row, col))
        used_rows.add(row)
        used_cols.add(col)
    return matched


def _clamp_cost(value: float) -> float:
    if value >= INF_COST:
        return INF_COST
    return max(0.0, min(float(value), 1e6))
