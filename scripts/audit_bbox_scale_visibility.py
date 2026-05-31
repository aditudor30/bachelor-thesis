"""Run GT bbox scale/visibility/difficulty audit."""

import argparse
from pathlib import Path
from typing import Any, Dict

from deep_oc_sort_3d.detection2d.yolo_bbox_audit import (
    audit_gt_bboxes,
    save_bbox_audit_csv,
    save_summary_json,
    summarize_bbox_audit,
)
from deep_oc_sort_3d.detection2d.yolo_bbox_difficulty import default_difficulty_config


def audit_bbox_scale_visibility(args: Any) -> None:
    """Run bbox audit and write CSV + JSON summary."""
    difficulty_config = _difficulty_config_from_args(args)
    records = audit_gt_bboxes(
        root=args.root,
        split=args.split,
        scenes=args.scenes,
        camera_id=args.camera_id,
        frame_stride=args.frame_stride,
        max_frames_per_scene=args.max_frames_per_scene,
        difficulty_config=difficulty_config,
    )
    summary = summarize_bbox_audit(records)
    save_bbox_audit_csv(records, args.output)
    save_summary_json(summary, args.summary_output)
    _print_summary(summary)
    print("Wrote CSV: %s" % args.output)
    print("Wrote summary: %s" % args.summary_output)


def _difficulty_config_from_args(args: Any) -> Dict[str, Any]:
    cfg = default_difficulty_config()
    if args.easy_area_threshold is not None:
        cfg["easy_area_norm"] = args.easy_area_threshold
    if args.medium_area_threshold is not None:
        cfg["medium_area_norm"] = args.medium_area_threshold
    if args.easy_min_side is not None:
        cfg["easy_min_side"] = args.easy_min_side
    if args.medium_min_side is not None:
        cfg["medium_min_side"] = args.medium_min_side
    return cfg


def _print_summary(summary: Dict[str, Any]) -> None:
    print("total boxes: %d" % int(summary.get("total_boxes", 0)))
    print("count per class: %s" % summary.get("count_per_class", {}))
    print("count per difficulty: %s" % summary.get("count_per_difficulty", {}))
    print("per-class median area_norm:")
    for class_name, stats in summary.get("per_class_stats", {}).items():
        print("  %s: %s" % (class_name, stats.get("area_norm_median")))
    print("top cameras per class:")
    for class_name, rows in summary.get("top_cameras_per_class", {}).items():
        print("  %s: %s" % (class_name, rows[:3]))
    print("top scenes per class:")
    for class_name, rows in summary.get("top_scenes_per_class", {}).items():
        print("  %s: %s" % (class_name, rows[:3]))
    if summary.get("warnings"):
        print("warnings: %s" % summary["warnings"])


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit GT bbox scale/visibility/difficulty.")
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--split", required=True, choices=["train", "val"])
    parser.add_argument("--scenes", nargs="+", required=True)
    parser.add_argument("--camera-id", default="all")
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--max-frames-per-scene", type=int, default=None)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--summary-output", required=True, type=Path)
    parser.add_argument("--easy-area-threshold", type=float, default=None)
    parser.add_argument("--medium-area-threshold", type=float, default=None)
    parser.add_argument("--easy-min-side", type=float, default=None)
    parser.add_argument("--medium-min-side", type=float, default=None)
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    audit_bbox_scale_visibility(args)


if __name__ == "__main__":
    main()

