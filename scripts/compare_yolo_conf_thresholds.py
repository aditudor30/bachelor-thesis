"""Compare YOLO precision/recall at multiple confidence thresholds."""

import argparse
import csv
from pathlib import Path
from typing import Any, Dict, List

from deep_oc_sort_3d.data.dataset_structure import get_scene_paths
from deep_oc_sort_3d.data.ground_truth import GroundTruthObject, load_ground_truth_json
from deep_oc_sort_3d.detection2d.yolo_detection_io import read_detections_csv
from deep_oc_sort_3d.detection2d.yolo_eval import evaluate_at_conf_thresholds, threshold_results_to_rows


CSV_FIELDS = [
    "threshold",
    "class_name",
    "detections",
    "gt_visible",
    "matches",
    "precision",
    "recall",
    "mean_iou",
]


def compare_yolo_conf_thresholds(args: Any) -> None:
    """Evaluate a confidence-threshold sweep."""
    detections = [
        det
        for det in read_detections_csv(args.detections)
        if det.split == args.split and det.scene_name == args.scene and det.camera_id == args.camera_id
        and (args.max_frames is None or int(det.frame_id) < int(args.max_frames))
    ]
    all_gt_objects_by_frame = _load_gt_by_frame(args.root, args.split, args.scene)
    evaluated_frame_ids = _resolve_evaluated_frame_ids(detections, args.max_frames, args.frame_stride)
    gt_objects_by_frame = _filter_gt_by_frame(all_gt_objects_by_frame, evaluated_frame_ids)
    if args.max_frames is None:
        print(
            "warning: --max-frames was not provided; threshold sweep uses frames present in the detection CSV. "
            "For recall over frames with no detections, pass the same --max-frames used for inference."
        )
    results = evaluate_at_conf_thresholds(
        detections=detections,
        gt_objects_by_frame=gt_objects_by_frame,
        camera_id=args.camera_id,
        thresholds=args.thresholds,
        iou_threshold=args.iou_threshold,
    )
    rows = threshold_results_to_rows(results)
    _write_csv(rows, args.output)
    print("Wrote %s" % args.output)
    recommendation = _recommend_threshold(rows, args.min_precision)
    print(
        "recommended_threshold: %s (min_precision=%.2f)"
        % ("None" if recommendation is None else "%.4f" % recommendation, args.min_precision)
    )


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


def _write_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _recommend_threshold(rows: List[Dict[str, Any]], min_precision: float) -> Any:
    overall = [row for row in rows if row["class_name"] == "all"]
    candidates = [row for row in overall if float(row["precision"]) >= float(min_precision)]
    if not candidates:
        candidates = overall
    if not candidates:
        return None
    best = sorted(candidates, key=lambda row: (float(row["recall"]), float(row["precision"])), reverse=True)[0]
    return float(best["threshold"])


def build_arg_parser() -> argparse.ArgumentParser:
    """Build CLI parser."""
    parser = argparse.ArgumentParser(description="Compare YOLO confidence thresholds.")
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--split", required=True, choices=["train", "val"])
    parser.add_argument("--scene", required=True)
    parser.add_argument("--detections", required=True, type=Path)
    parser.add_argument("--camera-id", required=True)
    parser.add_argument("--thresholds", nargs="+", type=float, required=True)
    parser.add_argument("--iou-threshold", type=float, default=0.3)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--min-precision", type=float, default=0.5)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    compare_yolo_conf_thresholds(args)


if __name__ == "__main__":
    main()
