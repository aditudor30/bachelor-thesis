"""Evaluate YOLO detections against visible GT boxes for a small frame window."""

import argparse
import csv
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from deep_oc_sort_3d.data.ground_truth import GroundTruthObject
from deep_oc_sort_3d.data.sample_dataset import SmartSpacesFrameDataset
from deep_oc_sort_3d.detection2d.yolo_detection_io import read_detections_csv
from deep_oc_sort_3d.detection2d.yolo_types import Detection2D
from deep_oc_sort_3d.observations.detection_gt_matching import match_detections_to_gt


CSV_FIELDS = [
    "frame_id",
    "camera_id",
    "class_name",
    "detection_count",
    "gt_visible_count",
    "matches",
    "precision",
    "recall",
    "mean_iou",
]


def evaluate_detection_gt_matching(args: Any) -> None:
    """Evaluate detection-to-GT matching and export per-frame/per-class rows."""
    if args.split == "test":
        print("test split has no ground truth; detection-GT matching is not available.")
        return

    detections = _load_filtered_detections(args)
    grouped = _group_detections(detections)
    cameras = _resolve_cameras(args, detections)
    frame_ids = _resolve_frame_ids(args, detections)

    rows = []
    dataset_cache = {}
    for camera_id in cameras:
        dataset_cache[camera_id] = SmartSpacesFrameDataset(
            root=args.root,
            split=args.split,
            scene_name=args.scene,
            max_frames=args.max_frames,
            camera_id=camera_id,
            load_rgb=False,
            load_depth=False,
            load_gt=True,
        )
        dataset = dataset_cache[camera_id]
        for frame_id in frame_ids:
            if frame_id < 0 or frame_id >= len(dataset):
                continue
            frame_detections = grouped.get((frame_id, camera_id), [])
            sample = dataset[frame_id]
            gt_objects = sample.get("gt_objects")
            if gt_objects is None:
                gt_objects = []
            rows.extend(
                _build_rows_for_frame(
                    frame_id=frame_id,
                    camera_id=camera_id,
                    detections=frame_detections,
                    gt_objects=gt_objects,
                    iou_threshold=args.iou_threshold,
                    class_must_match=args.class_must_match,
                )
            )

    _write_rows(rows, args.output)
    print("Wrote matching evaluation: %s" % args.output)
    _print_aggregate_summary(rows)


def _load_filtered_detections(args: Any) -> List[Detection2D]:
    detections = []
    for det in read_detections_csv(args.detections):
        if det.split != args.split or det.scene_name != args.scene:
            continue
        if args.camera_id is not None and det.camera_id != args.camera_id:
            continue
        if args.max_frames is not None and det.frame_id >= int(args.max_frames):
            continue
        detections.append(det)
    return detections


def _group_detections(detections: List[Detection2D]) -> Dict[Tuple[int, str], List[Detection2D]]:
    grouped = {}
    for det in detections:
        key = (int(det.frame_id), det.camera_id)
        if key not in grouped:
            grouped[key] = []
        grouped[key].append(det)
    return grouped


def _resolve_cameras(args: Any, detections: List[Detection2D]) -> List[str]:
    if args.camera_id is not None:
        return [args.camera_id]
    cameras = sorted(set(det.camera_id for det in detections))
    return cameras


def _resolve_frame_ids(args: Any, detections: List[Detection2D]) -> List[int]:
    if args.max_frames is not None:
        return list(range(int(args.max_frames)))
    return sorted(set(int(det.frame_id) for det in detections))


def _build_rows_for_frame(
    frame_id: int,
    camera_id: str,
    detections: List[Detection2D],
    gt_objects: List[GroundTruthObject],
    iou_threshold: float,
    class_must_match: bool,
) -> List[Dict[str, Any]]:
    matched_gt, matched_iou = match_detections_to_gt(
        detections=detections,
        gt_objects=gt_objects,
        camera_id=camera_id,
        iou_threshold=iou_threshold,
        class_must_match=class_must_match,
    )
    gt_visible = [obj for obj in gt_objects if camera_id in obj.visible_bboxes_2d]
    class_names = sorted(
        set([det.class_name for det in detections] + [obj.object_type for obj in gt_visible])
    )
    rows = []
    for class_name in class_names:
        det_count = len([det for det in detections if det.class_name == class_name])
        gt_count = len([obj for obj in gt_visible if obj.object_type == class_name])
        match_indices = [idx for idx, gt in matched_gt.items() if gt.object_type == class_name]
        match_count = len(match_indices)
        ious = [matched_iou[idx] for idx in match_indices if idx in matched_iou]
        precision = 0.0 if det_count == 0 else float(match_count) / float(det_count)
        recall = 0.0 if gt_count == 0 else float(match_count) / float(gt_count)
        rows.append(
            {
                "frame_id": frame_id,
                "camera_id": camera_id,
                "class_name": class_name,
                "detection_count": det_count,
                "gt_visible_count": gt_count,
                "matches": match_count,
                "precision": precision,
                "recall": recall,
                "mean_iou": None if not ious else float(np.mean(ious)),
            }
        )
    return rows


def _write_rows(rows: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _print_aggregate_summary(rows: List[Dict[str, Any]]) -> None:
    total_dets = sum(int(row["detection_count"]) for row in rows)
    total_gt = sum(int(row["gt_visible_count"]) for row in rows)
    total_matches = sum(int(row["matches"]) for row in rows)
    precision = 0.0 if total_dets == 0 else float(total_matches) / float(total_dets)
    recall = 0.0 if total_gt == 0 else float(total_matches) / float(total_gt)
    ious = [float(row["mean_iou"]) for row in rows if row["mean_iou"] is not None and row["matches"] > 0]
    print("Summary:")
    print("  detections: %d" % total_dets)
    print("  gt_visible: %d" % total_gt)
    print("  matches: %d" % total_matches)
    print("  precision: %.6f" % precision)
    print("  recall: %.6f" % recall)
    print("  mean_iou: %s" % (None if not ious else "%.6f" % float(np.mean(ious))))
    print("Per class:")
    for class_name in sorted(set(row["class_name"] for row in rows)):
        class_rows = [row for row in rows if row["class_name"] == class_name]
        class_dets = sum(int(row["detection_count"]) for row in class_rows)
        class_gt = sum(int(row["gt_visible_count"]) for row in class_rows)
        class_matches = sum(int(row["matches"]) for row in class_rows)
        class_precision = 0.0 if class_dets == 0 else float(class_matches) / float(class_dets)
        class_recall = 0.0 if class_gt == 0 else float(class_matches) / float(class_gt)
        print(
            "  %s: det=%d gt=%d matches=%d precision=%.4f recall=%.4f"
            % (class_name, class_dets, class_gt, class_matches, class_precision, class_recall)
        )


def build_arg_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser."""
    parser = argparse.ArgumentParser(description="Evaluate detection-GT matching.")
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--split", required=True, choices=["train", "val", "test"])
    parser.add_argument("--scene", required=True)
    parser.add_argument("--detections", required=True, type=Path)
    parser.add_argument("--camera-id", default=None)
    parser.add_argument("--max-frames", type=int, default=100)
    parser.add_argument("--iou-threshold", type=float, default=0.3)
    parser.add_argument("--output", required=True, type=Path)
    class_group = parser.add_mutually_exclusive_group()
    class_group.add_argument("--class-must-match", dest="class_must_match", action="store_true")
    class_group.add_argument("--no-class-must-match", dest="class_must_match", action="store_false")
    parser.set_defaults(class_must_match=True)
    return parser


def main() -> None:
    """CLI entrypoint."""
    parser = build_arg_parser()
    args = parser.parse_args()
    evaluate_detection_gt_matching(args)


if __name__ == "__main__":
    main()

