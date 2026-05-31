"""Evaluate YOLO detections on official validation scenes."""

import argparse
from pathlib import Path
from typing import Any, Optional

from deep_oc_sort_3d.detection2d.yolo_split_eval import (
    evaluate_yolo_dataset_predictions,
    save_metrics_csv,
    save_metrics_json,
    summarize_per_class_metrics,
)


def evaluate_yolo_official_val(args: Any) -> None:
    """Evaluate official val detections and save metrics."""
    metrics = evaluate_yolo_dataset_predictions(
        root=args.root,
        split="val",
        scenes=args.scenes,
        detections_dir=args.detections_dir,
        camera_id=_parse_camera_id(args.camera_id),
        iou_threshold=args.iou_threshold,
        conf_threshold=args.conf_threshold,
        max_frames_per_scene=args.max_frames_per_scene,
        frame_stride=args.frame_stride,
    )
    save_metrics_json(metrics, args.output)
    save_metrics_csv(metrics, args.output.with_suffix(".csv"))
    print("Official val may not contain all classes; inspect gt_visible per class.")
    print(summarize_per_class_metrics(metrics))
    print("Wrote %s" % args.output)
    print("Wrote %s" % args.output.with_suffix(".csv"))


def _parse_camera_id(value: str) -> Optional[str]:
    if value is None or str(value).lower() == "all":
        return None
    return str(value)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate YOLO on official val scenes.")
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--detections-dir", required=True, type=Path)
    parser.add_argument("--scenes", nargs="+", required=True)
    parser.add_argument("--camera-id", default="all")
    parser.add_argument("--iou-threshold", type=float, default=0.3)
    parser.add_argument("--conf-threshold", type=float, default=0.05)
    parser.add_argument("--max-frames-per-scene", type=int, default=None)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    evaluate_yolo_official_val(args)


if __name__ == "__main__":
    main()

