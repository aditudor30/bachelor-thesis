"""Audit visible GT class distribution for train/val scenes."""

import argparse
import csv
from pathlib import Path
from typing import Any, Dict, List

from deep_oc_sort_3d.detection2d.yolo_class_audit import (
    count_gt_visible_boxes_by_class,
    counts_to_csv_rows,
    find_frames_with_classes,
    summarize_class_distribution,
)


def audit_gt_classes(args: Any) -> None:
    """Print and optionally export GT class-count audit."""
    counts = count_gt_visible_boxes_by_class(
        root=args.root,
        split=args.split,
        scenes=args.scenes,
        camera_id=args.camera_id,
        max_frames_per_scene=args.max_frames_per_scene,
        frame_stride=args.frame_stride,
    )
    print(summarize_class_distribution(counts))
    print("")
    print("Per scene:")
    for scene_name, scene_stats in counts.get("per_scene", {}).items():
        print("  %s: %s" % (scene_name, scene_stats.get("class_counts", {})))
    print("")
    print("Per camera:")
    for camera_id, class_counts in counts.get("per_camera", {}).items():
        print("  %s: %s" % (camera_id, class_counts))

    rare_frames = find_frames_with_classes(
        root=args.root,
        split=args.split,
        scenes=args.scenes,
        target_classes=args.target_classes,
        camera_id=args.camera_id,
        max_frames_per_scene=args.max_frames_per_scene,
        frame_stride=args.frame_stride,
    )
    print("")
    print("Frames with target classes (%s): %d" % (", ".join(args.target_classes), len(rare_frames)))
    for item in rare_frames[:10]:
        print(
            "  %s %s frame=%d counts=%s"
            % (item["scene_name"], item["camera_id"], item["frame_id"], item["target_class_counts"])
        )

    if args.output is not None:
        _write_csv(counts_to_csv_rows(counts), args.output)
        print("Wrote %s" % args.output)


def _write_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["scope", "name", "class_name", "count"])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def build_arg_parser() -> argparse.ArgumentParser:
    """Build CLI parser."""
    parser = argparse.ArgumentParser(description="Audit visible GT class distribution.")
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--split", required=True, choices=["train", "val"])
    parser.add_argument("--scenes", nargs="+", required=True)
    parser.add_argument("--camera-id", default=None)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--max-frames-per-scene", type=int, default=None)
    parser.add_argument("--target-classes", nargs="+", default=["Forklift", "PalletTruck"])
    parser.add_argument("--output", type=Path, default=None)
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    audit_gt_classes(args)


if __name__ == "__main__":
    main()

