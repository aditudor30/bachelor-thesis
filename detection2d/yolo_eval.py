"""Per-class evaluation for YOLO detections against visible GT boxes."""

from typing import Any, Dict, List, Tuple

import numpy as np

from deep_oc_sort_3d.data.ground_truth import GroundTruthObject
from deep_oc_sort_3d.detection2d.yolo_types import Detection2D
from deep_oc_sort_3d.observations.detection_gt_matching import DEFAULT_CLASS_MAPPING, match_detections_to_gt


def evaluate_detections_vs_gt(
    detections: List[Detection2D],
    gt_objects_by_frame: Dict[int, List[GroundTruthObject]],
    camera_id: str,
    iou_threshold: float = 0.3,
    class_must_match: bool = True,
) -> Dict[str, Any]:
    """Evaluate detections against visible GT boxes for one camera."""
    detections_by_frame = _group_detections_by_frame(detections)
    frame_ids = sorted(set(list(detections_by_frame.keys()) + list(gt_objects_by_frame.keys())))
    stats = _empty_stats()
    all_ious = []
    for frame_id in frame_ids:
        frame_dets = detections_by_frame.get(frame_id, [])
        frame_gt = gt_objects_by_frame.get(frame_id, [])
        visible_gt = [obj for obj in frame_gt if camera_id in obj.visible_bboxes_2d]
        matched_gt, matched_iou = match_detections_to_gt(
            frame_dets,
            frame_gt,
            camera_id,
            iou_threshold=iou_threshold,
            class_must_match=class_must_match,
        )

        stats["detections"] += len(frame_dets)
        stats["gt_visible"] += len(visible_gt)
        stats["matches"] += len(matched_gt)
        all_ious.extend(list(matched_iou.values()))

        for det in frame_dets:
            _ensure_class(stats, det.class_name)
            stats["per_class"][det.class_name]["detections"] += 1
        for gt in visible_gt:
            _ensure_class(stats, gt.object_type)
            stats["per_class"][gt.object_type]["gt_visible"] += 1
        for det_idx, gt in matched_gt.items():
            _ensure_class(stats, gt.object_type)
            class_stats = stats["per_class"][gt.object_type]
            class_stats["matches"] += 1
            if det_idx in matched_iou:
                class_stats["ious"].append(float(matched_iou[det_idx]))

    return _finalize_stats(stats, all_ious)


def evaluate_at_conf_thresholds(
    detections: List[Detection2D],
    gt_objects_by_frame: Dict[int, List[GroundTruthObject]],
    camera_id: str,
    thresholds: List[float],
    iou_threshold: float = 0.3,
) -> Dict[str, Any]:
    """Evaluate the same detection set at multiple confidence thresholds."""
    results = {}
    for threshold in thresholds:
        filtered = [det for det in detections if float(det.confidence) >= float(threshold)]
        results[float(threshold)] = evaluate_detections_vs_gt(
            detections=filtered,
            gt_objects_by_frame=gt_objects_by_frame,
            camera_id=camera_id,
            iou_threshold=iou_threshold,
            class_must_match=True,
        )
    return {
        "thresholds": [float(value) for value in thresholds],
        "results": results,
    }


def _group_detections_by_frame(detections: List[Detection2D]) -> Dict[int, List[Detection2D]]:
    grouped = {}
    for det in detections:
        frame_id = int(det.frame_id)
        if frame_id not in grouped:
            grouped[frame_id] = []
        grouped[frame_id].append(det)
    return grouped


def _empty_stats() -> Dict[str, Any]:
    per_class = {}
    for class_name in DEFAULT_CLASS_MAPPING.keys():
        per_class[class_name] = {
            "detections": 0,
            "gt_visible": 0,
            "matches": 0,
            "ious": [],
        }
    return {
        "detections": 0,
        "gt_visible": 0,
        "matches": 0,
        "per_class": per_class,
    }


def _ensure_class(stats: Dict[str, Any], class_name: str) -> None:
    if class_name not in stats["per_class"]:
        stats["per_class"][class_name] = {
            "detections": 0,
            "gt_visible": 0,
            "matches": 0,
            "ious": [],
        }


def _finalize_stats(stats: Dict[str, Any], all_ious: List[float]) -> Dict[str, Any]:
    detections = int(stats["detections"])
    gt_visible = int(stats["gt_visible"])
    matches = int(stats["matches"])
    finalized = {
        "detections": detections,
        "gt_visible": gt_visible,
        "matches": matches,
        "precision": 0.0 if detections == 0 else float(matches) / float(detections),
        "recall": 0.0 if gt_visible == 0 else float(matches) / float(gt_visible),
        "mean_iou": None if not all_ious else float(np.mean(all_ious)),
        "per_class": {},
    }
    for class_name, class_stats in stats["per_class"].items():
        class_dets = int(class_stats["detections"])
        class_gt = int(class_stats["gt_visible"])
        class_matches = int(class_stats["matches"])
        class_ious = class_stats["ious"]
        finalized["per_class"][class_name] = {
            "detections": class_dets,
            "gt_visible": class_gt,
            "matches": class_matches,
            "precision": 0.0 if class_dets == 0 else float(class_matches) / float(class_dets),
            "recall": 0.0 if class_gt == 0 else float(class_matches) / float(class_gt),
            "mean_iou": None if not class_ious else float(np.mean(class_ious)),
        }
    return finalized


def threshold_results_to_rows(results: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Flatten threshold sweep metrics into CSV-friendly rows."""
    rows = []
    for threshold in results.get("thresholds", []):
        metrics = results.get("results", {}).get(float(threshold), {})
        rows.append(_metric_row(float(threshold), "all", metrics))
        for class_name, class_metrics in metrics.get("per_class", {}).items():
            rows.append(_metric_row(float(threshold), class_name, class_metrics))
    return rows


def _metric_row(threshold: float, class_name: str, metrics: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "threshold": threshold,
        "class_name": class_name,
        "detections": int(metrics.get("detections", 0)),
        "gt_visible": int(metrics.get("gt_visible", 0)),
        "matches": int(metrics.get("matches", 0)),
        "precision": float(metrics.get("precision", 0.0)),
        "recall": float(metrics.get("recall", 0.0)),
        "mean_iou": metrics.get("mean_iou"),
    }

