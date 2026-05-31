"""Evaluate YOLO detections against GT visible boxes per class."""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from deep_oc_sort_3d.data.dataset_structure import get_scene_paths
from deep_oc_sort_3d.data.ground_truth import GroundTruthObject, load_ground_truth_json
from deep_oc_sort_3d.detection2d.yolo_detection_io import read_detections_csv
from deep_oc_sort_3d.detection2d.yolo_eval import evaluate_detections_vs_gt


def evaluate_yolo_per_class(args: Any) -> None:
    """Filter detections, evaluate per class, and write JSON."""
    detections = [
        det
        for det in read_detections_csv(args.detections)
        if det.split == args.split
        and det.scene_name == args.scene
        and det.camera_id == args.camera_id
        and float(det.confidence) >= float(args.conf_threshold)
        and (args.max_frames is None or int(det.frame_id) < int(args.max_frames))
    ]
    all_gt_objects_by_frame = _load_gt_by_frame(args.root, args.split, args.scene)
    evaluated_frame_ids = _resolve_evaluated_frame_ids(detections, args.max_frames, args.frame_stride)
    gt_objects_by_frame = _filter_gt_by_frame(all_gt_objects_by_frame, evaluated_frame_ids)
    if args.max_frames is None:
        print(
            "warning: --max-frames was not provided; evaluation uses frames present in the detection CSV. "
            "For recall over frames with no detections, pass the same --max-frames used for inference."
        )
    metrics = evaluate_detections_vs_gt(
        detections=detections,
        gt_objects_by_frame=gt_objects_by_frame,
        camera_id=args.camera_id,
        iou_threshold=args.iou_threshold,
        class_must_match=args.class_must_match,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    print("Wrote %s" % args.output)
    _print_summary(metrics)


def _load_gt_by_frame(root: Path, split: str, scene: str) -> Dict[int, List[GroundTruthObject]]:
    scene_paths = get_scene_paths(root, split, scene)
    if scene_paths.ground_truth_path is None or not scene_paths.ground_truth_path.exists():
        return {}
    objects = load_ground_truth_json(scene_paths.ground_truth_path)
    grouped = {}
    for obj in objects:
        frame_id = int(obj.frame_id)
        if frame_id not in grouped:
            grouped[frame_id] = []
        grouped[frame_id].append(obj)
    return grouped


def _resolve_evaluated_frame_ids(detections: List[Any], max_frames: Any, frame_stride: int) -> List[int]:
    if max_frames is not None:
        return list(range(0, int(max_frames), max(int(frame_stride), 1)))
    return sorted(set(int(det.frame_id) for det in detections))


def _filter_gt_by_frame(
    gt_objects_by_frame: Dict[int, List[GroundTruthObject]],
    frame_ids: List[int],
) -> Dict[int, List[GroundTruthObject]]:
    selected = {}
    frame_set = set(int(frame_id) for frame_id in frame_ids)
    for frame_id, objects in gt_objects_by_frame.items():
        if int(frame_id) in frame_set:
            selected[int(frame_id)] = objects
    for frame_id in frame_set:
        if frame_id not in selected:
            selected[frame_id] = []
    return selected


def _print_summary(metrics: Dict[str, Any]) -> None:
    print("Summary:")
    print("  detections: %d" % int(metrics.get("detections", 0)))
    print("  gt_visible: %d" % int(metrics.get("gt_visible", 0)))
    print("  matches: %d" % int(metrics.get("matches", 0)))
    print("  precision: %.6f" % float(metrics.get("precision", 0.0)))
    print("  recall: %.6f" % float(metrics.get("recall", 0.0)))
    print("  mean_iou: %s" % metrics.get("mean_iou"))
    print("Per class:")
    for class_name, class_metrics in metrics.get("per_class", {}).items():
        print(
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


def build_arg_parser() -> argparse.ArgumentParser:
    """Build CLI parser."""
    parser = argparse.ArgumentParser(description="Evaluate YOLO detections per class.")
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--split", required=True, choices=["train", "val"])
    parser.add_argument("--scene", required=True)
    parser.add_argument("--detections", required=True, type=Path)
    parser.add_argument("--camera-id", required=True)
    parser.add_argument("--iou-threshold", type=float, default=0.3)
    parser.add_argument("--conf-threshold", type=float, default=0.05)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--output", required=True, type=Path)
    class_group = parser.add_mutually_exclusive_group()
    class_group.add_argument("--class-must-match", dest="class_must_match", action="store_true")
    class_group.add_argument("--no-class-must-match", dest="class_must_match", action="store_false")
    parser.set_defaults(class_must_match=True)
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    evaluate_yolo_per_class(args)


if __name__ == "__main__":
    main()
