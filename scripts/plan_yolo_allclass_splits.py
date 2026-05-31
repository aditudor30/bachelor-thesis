"""Plan all-class YOLO train, internal holdout, and official val splits."""

import argparse
import json
from pathlib import Path
from typing import Any

from deep_oc_sort_3d.detection2d.yolo_split_planner import (
    audit_scene_class_coverage,
    build_split_plan_dict,
    summarize_split_coverage,
    validate_no_scene_overlap,
)


def plan_yolo_allclass_splits(args: Any) -> None:
    """Audit proposed splits and write a JSON plan."""
    train_coverage = audit_scene_class_coverage(
        args.root,
        "train",
        args.train_candidates,
        camera_id=args.camera_id,
        frame_stride=args.frame_stride,
        max_frames_per_scene=args.max_frames_per_scene,
    )
    holdout_coverage = audit_scene_class_coverage(
        args.root,
        "train",
        args.holdout_scenes,
        camera_id=args.camera_id,
        frame_stride=args.frame_stride,
        max_frames_per_scene=args.max_frames_per_scene,
    )
    val_coverage = audit_scene_class_coverage(
        args.root,
        "val",
        args.official_val_scenes,
        camera_id=args.camera_id,
        frame_stride=args.frame_stride,
        max_frames_per_scene=args.max_frames_per_scene,
    )
    overlap_ok = validate_no_scene_overlap(args.train_candidates, args.holdout_scenes, args.official_val_scenes)
    plan = build_split_plan_dict(
        train_scenes=args.train_candidates,
        holdout_scenes=args.holdout_scenes,
        official_val_scenes=args.official_val_scenes,
        train_coverage=train_coverage,
        holdout_coverage=holdout_coverage,
        official_val_coverage=val_coverage,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(plan, indent=2, sort_keys=True), encoding="utf-8")
    print("no_scene_overlap: %s" % overlap_ok)
    print(summarize_split_coverage({"train": train_coverage, "internal_holdout": holdout_coverage, "official_val": val_coverage}))
    print("Wrote %s" % args.output)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plan YOLO all-class scene splits.")
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--train-candidates", nargs="+", required=True)
    parser.add_argument("--holdout-scenes", nargs="+", required=True)
    parser.add_argument("--official-val-scenes", nargs="+", required=True)
    parser.add_argument("--camera-id", default=None)
    parser.add_argument("--frame-stride", type=int, default=5)
    parser.add_argument("--max-frames-per-scene", type=int, default=1000)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    plan_yolo_allclass_splits(args)


if __name__ == "__main__":
    main()

