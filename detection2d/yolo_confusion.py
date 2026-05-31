"""Confusion diagnostics for YOLO detections."""

from typing import Any, Dict, List, Optional, Tuple

from deep_oc_sort_3d.data.ground_truth import GroundTruthObject
from deep_oc_sort_3d.detection2d.yolo_types import Detection2D
from deep_oc_sort_3d.observations.detection_gt_matching import match_detections_to_gt


def compute_confusion_matrix_from_matches(
    detections: List[Detection2D],
    gt_objects_by_frame: Dict[int, List[GroundTruthObject]],
    camera_id: str,
    iou_threshold: float = 0.3,
    conf_threshold: float = 0.0,
) -> Dict[str, Dict[str, int]]:
    """Compute GT-class to predicted-class confusion matrix for IoU matches."""
    matrix = {}
    for frame_id, frame_detections, frame_gt in _iter_frames(detections, gt_objects_by_frame, conf_threshold):
        matched_gt, _matched_iou = match_detections_to_gt(
            frame_detections,
            frame_gt,
            camera_id,
            iou_threshold=iou_threshold,
            class_must_match=False,
        )
        for det_idx, gt in matched_gt.items():
            pred = frame_detections[det_idx].class_name
            gt_name = gt.object_type
            if gt_name not in matrix:
                matrix[gt_name] = {}
            matrix[gt_name][pred] = matrix[gt_name].get(pred, 0) + 1
    return matrix


def collect_false_positives(
    detections: List[Detection2D],
    gt_objects_by_frame: Dict[int, List[GroundTruthObject]],
    camera_id: str,
    iou_threshold: float = 0.3,
    conf_threshold: float = 0.0,
) -> List[Dict[str, Any]]:
    """Collect detections that do not match any visible GT object."""
    rows = []
    for frame_id, frame_detections, frame_gt in _iter_frames(detections, gt_objects_by_frame, conf_threshold):
        matched_gt, _matched_iou = match_detections_to_gt(
            frame_detections,
            frame_gt,
            camera_id,
            iou_threshold=iou_threshold,
            class_must_match=False,
        )
        matched_indices = set(matched_gt.keys())
        for det_idx, det in enumerate(frame_detections):
            if det_idx in matched_indices:
                continue
            rows.append(_detection_row(det, "false_positive", None, None))
    return rows


def collect_false_negatives(
    detections: List[Detection2D],
    gt_objects_by_frame: Dict[int, List[GroundTruthObject]],
    camera_id: str,
    iou_threshold: float = 0.3,
    conf_threshold: float = 0.0,
) -> List[Dict[str, Any]]:
    """Collect visible GT objects not matched by any detection."""
    rows = []
    for frame_id, frame_detections, frame_gt in _iter_frames(detections, gt_objects_by_frame, conf_threshold):
        visible_gt = [obj for obj in frame_gt if camera_id in obj.visible_bboxes_2d]
        matched_gt, _matched_iou = match_detections_to_gt(
            frame_detections,
            frame_gt,
            camera_id,
            iou_threshold=iou_threshold,
            class_must_match=False,
        )
        matched_object_ids = set(gt.object_id for gt in matched_gt.values())
        for gt in visible_gt:
            if gt.object_id in matched_object_ids:
                continue
            rows.append(_gt_row(gt, camera_id, "false_negative", None, None))
    return rows


def collect_class_confusions(
    detections: List[Detection2D],
    gt_objects_by_frame: Dict[int, List[GroundTruthObject]],
    camera_id: str,
    iou_threshold: float = 0.3,
    conf_threshold: float = 0.0,
) -> List[Dict[str, Any]]:
    """Collect matched detections where predicted class differs from GT class."""
    rows = []
    for frame_id, frame_detections, frame_gt in _iter_frames(detections, gt_objects_by_frame, conf_threshold):
        matched_gt, matched_iou = match_detections_to_gt(
            frame_detections,
            frame_gt,
            camera_id,
            iou_threshold=iou_threshold,
            class_must_match=False,
        )
        for det_idx, gt in matched_gt.items():
            det = frame_detections[det_idx]
            if det.class_name == gt.object_type:
                continue
            rows.append(_detection_row(det, "class_confusion", gt, matched_iou.get(det_idx)))
    return rows


def summarize_confusions(confusions: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Summarize class confusion rows by GT/pred pair."""
    pair_counts = {}
    for item in confusions:
        pair = "%s->%s" % (item.get("gt_class_name"), item.get("pred_class_name"))
        pair_counts[pair] = pair_counts.get(pair, 0) + 1
    return {
        "num_confusions": len(confusions),
        "pair_counts": pair_counts,
        "top_pairs": sorted(pair_counts.items(), key=lambda item: item[1], reverse=True),
    }


def _iter_frames(
    detections: List[Detection2D],
    gt_objects_by_frame: Dict[int, List[GroundTruthObject]],
    conf_threshold: float,
) -> List[Tuple[int, List[Detection2D], List[GroundTruthObject]]]:
    detections_by_frame = {}
    for det in detections:
        if float(det.confidence) < float(conf_threshold):
            continue
        detections_by_frame.setdefault(int(det.frame_id), []).append(det)
    frame_ids = sorted(set(list(detections_by_frame.keys()) + list(gt_objects_by_frame.keys())))
    rows = []
    for frame_id in frame_ids:
        rows.append((frame_id, detections_by_frame.get(frame_id, []), gt_objects_by_frame.get(frame_id, [])))
    return rows


def _detection_row(
    det: Detection2D,
    error_type: str,
    gt: Optional[GroundTruthObject],
    iou: Optional[float],
) -> Dict[str, Any]:
    return {
        "type": error_type,
        "scene_name": det.scene_name,
        "split": det.split,
        "camera_id": det.camera_id,
        "frame_id": det.frame_id,
        "pred_class_name": det.class_name,
        "pred_class_id": det.class_id,
        "confidence": det.confidence,
        "bbox_xyxy": list(det.bbox_xyxy),
        "gt_class_name": None if gt is None else gt.object_type,
        "gt_object_id": None if gt is None else gt.object_id,
        "gt_bbox_xyxy": None if gt is None or det.camera_id not in gt.visible_bboxes_2d else list(gt.visible_bboxes_2d[det.camera_id]),
        "iou": iou,
    }


def _gt_row(
    gt: GroundTruthObject,
    camera_id: str,
    error_type: str,
    pred: Optional[Detection2D],
    iou: Optional[float],
) -> Dict[str, Any]:
    return {
        "type": error_type,
        "scene_name": None,
        "split": None,
        "camera_id": camera_id,
        "frame_id": gt.frame_id,
        "pred_class_name": None if pred is None else pred.class_name,
        "pred_class_id": None if pred is None else pred.class_id,
        "confidence": None if pred is None else pred.confidence,
        "bbox_xyxy": None if pred is None else list(pred.bbox_xyxy),
        "gt_class_name": gt.object_type,
        "gt_object_id": gt.object_id,
        "gt_bbox_xyxy": list(gt.visible_bboxes_2d[camera_id]),
        "iou": iou,
    }

