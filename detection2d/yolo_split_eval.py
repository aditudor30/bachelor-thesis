"""Evaluate YOLO detections over official or internal splits."""

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from deep_oc_sort_3d.data.dataset_structure import get_scene_paths
from deep_oc_sort_3d.data.ground_truth import GroundTruthObject, load_ground_truth_json
from deep_oc_sort_3d.detection2d.yolo_detection_io import read_detections_csv
from deep_oc_sort_3d.detection2d.yolo_eval import evaluate_detections_vs_gt
from deep_oc_sort_3d.detection2d.yolo_types import Detection2D


def evaluate_yolo_dataset_predictions(
    root: Union[str, Path],
    split: str,
    scenes: List[str],
    detections_dir: Union[str, Path],
    camera_id: Optional[str] = None,
    iou_threshold: float = 0.3,
    conf_threshold: float = 0.05,
    max_frames_per_scene: Optional[int] = None,
    frame_stride: int = 1,
) -> Dict[str, Any]:
    """Evaluate detections from a directory of CSV files over a scene split."""
    detections = _load_detection_dir(detections_dir)
    aggregate = _empty_aggregate()
    per_scene = {}
    for scene_name in scenes:
        scene_detections = [
            det
            for det in detections
            if det.split == split
            and det.scene_name == scene_name
            and float(det.confidence) >= float(conf_threshold)
            and (camera_id is None or det.camera_id == camera_id)
            and _frame_is_selected(det.frame_id, max_frames_per_scene, frame_stride)
        ]
        cameras = _resolve_cameras(scene_detections, root, split, scene_name, camera_id)
        scene_metrics = _empty_aggregate()
        gt_by_frame = _load_gt_by_frame(Path(root), split, scene_name, max_frames_per_scene, frame_stride)
        for cam in cameras:
            cam_detections = [det for det in scene_detections if det.camera_id == cam]
            cam_metrics = evaluate_detections_vs_gt(
                detections=cam_detections,
                gt_objects_by_frame=gt_by_frame,
                camera_id=cam,
                iou_threshold=iou_threshold,
                class_must_match=True,
            )
            _accumulate_metrics(scene_metrics, cam_metrics)
            _accumulate_metrics(aggregate, cam_metrics)
        per_scene[scene_name] = _finalize_aggregate(scene_metrics)
    metrics = _finalize_aggregate(aggregate)
    metrics["split"] = split
    metrics["scenes"] = list(scenes)
    metrics["camera_id"] = camera_id
    metrics["per_scene"] = per_scene
    return metrics


def summarize_per_class_metrics(metrics: Dict[str, Any]) -> str:
    """Create a readable per-class metric summary."""
    lines = []
    lines.append("detections: %d" % int(metrics.get("detections", 0)))
    lines.append("gt_visible: %d" % int(metrics.get("gt_visible", 0)))
    lines.append("matches: %d" % int(metrics.get("matches", 0)))
    lines.append("precision: %.6f" % float(metrics.get("precision", 0.0)))
    lines.append("recall: %.6f" % float(metrics.get("recall", 0.0)))
    lines.append("mean_iou: %s" % metrics.get("mean_iou"))
    lines.append("Per class:")
    for class_name, class_metrics in metrics.get("per_class", {}).items():
        lines.append(
            "  %s: det=%d gt=%d matches=%d precision=%.4f recall=%.4f"
            % (
                class_name,
                int(class_metrics.get("detections", 0)),
                int(class_metrics.get("gt_visible", 0)),
                int(class_metrics.get("matches", 0)),
                float(class_metrics.get("precision", 0.0)),
                float(class_metrics.get("recall", 0.0)),
            )
        )
    return "\n".join(lines)


def save_metrics_json(metrics: Dict[str, Any], path: Union[str, Path]) -> None:
    """Save metrics as JSON."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")


def save_metrics_csv(metrics: Dict[str, Any], path: Union[str, Path]) -> None:
    """Save overall and per-class metrics as CSV."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fields = ["class_name", "detections", "gt_visible", "matches", "precision", "recall", "mean_iou"]
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerow(_metric_row("all", metrics))
        for class_name, class_metrics in metrics.get("per_class", {}).items():
            writer.writerow(_metric_row(class_name, class_metrics))


def _load_detection_dir(path: Union[str, Path]) -> List[Detection2D]:
    detections = []
    directory = Path(path)
    if directory.is_file():
        return read_detections_csv(directory)
    for csv_path in sorted(directory.rglob("*.csv")):
        detections.extend(read_detections_csv(csv_path))
    return detections


def _load_gt_by_frame(
    root: Path,
    split: str,
    scene_name: str,
    max_frames_per_scene: Optional[int],
    frame_stride: int,
) -> Dict[int, List[GroundTruthObject]]:
    scene_paths = get_scene_paths(root, split, scene_name)
    if scene_paths.ground_truth_path is None or not scene_paths.ground_truth_path.exists():
        return {}
    objects = load_ground_truth_json(scene_paths.ground_truth_path)
    grouped = {}
    max_frame = None if max_frames_per_scene is None else int(max_frames_per_scene)
    stride = max(int(frame_stride), 1)
    for obj in objects:
        if max_frame is not None and int(obj.frame_id) >= max_frame:
            continue
        if int(obj.frame_id) % stride != 0:
            continue
        frame_id = int(obj.frame_id)
        if frame_id not in grouped:
            grouped[frame_id] = []
        grouped[frame_id].append(obj)
    if max_frame is not None:
        for frame_id in range(0, max_frame, stride):
            if frame_id not in grouped:
                grouped[frame_id] = []
    return grouped


def _resolve_cameras(
    detections: List[Detection2D],
    root: Union[str, Path],
    split: str,
    scene_name: str,
    camera_id: Optional[str],
) -> List[str]:
    if camera_id is not None:
        return [camera_id]
    cameras = set(det.camera_id for det in detections)
    scene_paths = get_scene_paths(Path(root), split, scene_name)
    if scene_paths.ground_truth_path is not None and scene_paths.ground_truth_path.exists():
        for obj in load_ground_truth_json(scene_paths.ground_truth_path):
            cameras.update(obj.visible_bboxes_2d.keys())
    return sorted(cameras)


def _frame_is_selected(frame_id: int, max_frames_per_scene: Optional[int], frame_stride: int) -> bool:
    if max_frames_per_scene is not None and int(frame_id) >= int(max_frames_per_scene):
        return False
    return int(frame_id) % max(int(frame_stride), 1) == 0


def _empty_aggregate() -> Dict[str, Any]:
    return {
        "detections": 0,
        "gt_visible": 0,
        "matches": 0,
        "weighted_iou_sum": 0.0,
        "per_class": {},
    }


def _accumulate_metrics(total: Dict[str, Any], metrics: Dict[str, Any]) -> None:
    matches = int(metrics.get("matches", 0))
    total["detections"] += int(metrics.get("detections", 0))
    total["gt_visible"] += int(metrics.get("gt_visible", 0))
    total["matches"] += matches
    if metrics.get("mean_iou") is not None:
        total["weighted_iou_sum"] += float(metrics["mean_iou"]) * float(matches)
    for class_name, class_metrics in metrics.get("per_class", {}).items():
        if class_name not in total["per_class"]:
            total["per_class"][class_name] = _empty_aggregate()
        _accumulate_metrics(total["per_class"][class_name], class_metrics)


def _finalize_aggregate(total: Dict[str, Any]) -> Dict[str, Any]:
    detections = int(total.get("detections", 0))
    gt_visible = int(total.get("gt_visible", 0))
    matches = int(total.get("matches", 0))
    finalized = {
        "detections": detections,
        "gt_visible": gt_visible,
        "matches": matches,
        "precision": 0.0 if detections == 0 else float(matches) / float(detections),
        "recall": 0.0 if gt_visible == 0 else float(matches) / float(gt_visible),
        "mean_iou": None if matches == 0 else float(total.get("weighted_iou_sum", 0.0)) / float(matches),
        "per_class": {},
    }
    for class_name, class_total in total.get("per_class", {}).items():
        finalized["per_class"][class_name] = _finalize_aggregate(class_total)
    return finalized


def _metric_row(class_name: str, metrics: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "class_name": class_name,
        "detections": int(metrics.get("detections", 0)),
        "gt_visible": int(metrics.get("gt_visible", 0)),
        "matches": int(metrics.get("matches", 0)),
        "precision": float(metrics.get("precision", 0.0)),
        "recall": float(metrics.get("recall", 0.0)),
        "mean_iou": metrics.get("mean_iou"),
    }

