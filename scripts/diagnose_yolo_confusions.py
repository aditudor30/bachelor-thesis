"""Diagnose YOLO false positives, false negatives, and class confusions."""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from deep_oc_sort_3d.data.dataset_structure import get_scene_paths
from deep_oc_sort_3d.data.ground_truth import GroundTruthObject, load_ground_truth_json
from deep_oc_sort_3d.detection2d.yolo_confusion import (
    collect_class_confusions,
    collect_false_negatives,
    collect_false_positives,
    compute_confusion_matrix_from_matches,
    summarize_confusions,
)
from deep_oc_sort_3d.detection2d.yolo_detection_io import read_detections_csv
from deep_oc_sort_3d.detection2d.yolo_types import Detection2D


def diagnose_yolo_confusions(args: Any) -> None:
    """Compute and save confusion diagnostics."""
    detections = _load_detection_dir(args.detections_dir)
    camera_filter = _parse_camera_id(args.camera_id)
    result = {
        "split": args.split,
        "scenes": args.scenes,
        "camera_id": args.camera_id,
        "confusion_matrix": {},
        "false_positives": [],
        "false_negatives": [],
        "class_confusions": [],
    }
    for scene_name in args.scenes:
        scene_dets = [
            det
            for det in detections
            if det.split == args.split
            and det.scene_name == scene_name
            and float(det.confidence) >= float(args.conf_threshold)
            and (camera_filter is None or det.camera_id == camera_filter)
        ]
        gt_by_frame = _load_gt_by_frame(args.root, args.split, scene_name)
        cameras = _resolve_cameras(scene_dets, gt_by_frame, camera_filter)
        for cam in cameras:
            cam_dets = [det for det in scene_dets if det.camera_id == cam]
            matrix = compute_confusion_matrix_from_matches(cam_dets, gt_by_frame, cam, args.iou_threshold)
            _merge_matrix(result["confusion_matrix"], matrix)
            false_positives = collect_false_positives(cam_dets, gt_by_frame, cam, args.iou_threshold)
            false_negatives = collect_false_negatives(cam_dets, gt_by_frame, cam, args.iou_threshold)
            class_confusions = collect_class_confusions(cam_dets, gt_by_frame, cam, args.iou_threshold)
            for row in false_positives + false_negatives + class_confusions:
                row["scene_name"] = scene_name
                row["split"] = args.split
            result["false_positives"].extend(false_positives)
            result["false_negatives"].extend(false_negatives)
            result["class_confusions"].extend(class_confusions)

    result["summary"] = {
        "false_positives": len(result["false_positives"]),
        "false_negatives": len(result["false_negatives"]),
        "class_confusions": summarize_confusions(result["class_confusions"]),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print("Wrote %s" % args.output)
    print("false_positives: %d" % len(result["false_positives"]))
    print("false_negatives: %d" % len(result["false_negatives"]))
    print("top confusion pairs: %s" % result["summary"]["class_confusions"]["top_pairs"][:10])


def _load_detection_dir(path: Path) -> List[Detection2D]:
    if path.is_file():
        return read_detections_csv(path)
    detections = []
    for csv_path in sorted(path.rglob("*.csv")):
        detections.extend(read_detections_csv(csv_path))
    return detections


def _load_gt_by_frame(root: Path, split: str, scene_name: str) -> Dict[int, List[GroundTruthObject]]:
    scene_paths = get_scene_paths(root, split, scene_name)
    if scene_paths.ground_truth_path is None or not scene_paths.ground_truth_path.exists():
        return {}
    grouped = {}
    for obj in load_ground_truth_json(scene_paths.ground_truth_path):
        grouped.setdefault(int(obj.frame_id), []).append(obj)
    return grouped


def _resolve_cameras(
    detections: List[Detection2D],
    gt_by_frame: Dict[int, List[GroundTruthObject]],
    camera_id: Optional[str],
) -> List[str]:
    if camera_id is not None:
        return [camera_id]
    cameras = set(det.camera_id for det in detections)
    for objects in gt_by_frame.values():
        for obj in objects:
            cameras.update(obj.visible_bboxes_2d.keys())
    return sorted(cameras)


def _merge_matrix(dst: Dict[str, Dict[str, int]], src: Dict[str, Dict[str, int]]) -> None:
    for gt_name, pred_counts in src.items():
        if gt_name not in dst:
            dst[gt_name] = {}
        for pred_name, count in pred_counts.items():
            dst[gt_name][pred_name] = dst[gt_name].get(pred_name, 0) + int(count)


def _parse_camera_id(value: str) -> Optional[str]:
    if value is None or str(value).lower() == "all":
        return None
    return str(value)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Diagnose YOLO class confusions.")
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--split", required=True, choices=["train", "val"])
    parser.add_argument("--scenes", nargs="+", required=True)
    parser.add_argument("--detections-dir", required=True, type=Path)
    parser.add_argument("--camera-id", default="all")
    parser.add_argument("--iou-threshold", type=float, default=0.3)
    parser.add_argument("--conf-threshold", type=float, default=0.05)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    diagnose_yolo_confusions(args)


if __name__ == "__main__":
    main()

