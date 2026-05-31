"""Export a balanced YOLO dataset with extra rare-class coverage."""

import argparse
from pathlib import Path
from typing import Any, Dict, List

from deep_oc_sort_3d.detection2d.yolo_balanced_exporter import BalancedYoloDatasetExporter
from deep_oc_sort_3d.detection2d.yolo_class_audit import (
    count_gt_visible_boxes_by_class,
    summarize_class_distribution,
)


def export_yolo_dataset_balanced(args: Any) -> None:
    """Export train/val balanced YOLO data."""
    exporter = BalancedYoloDatasetExporter(
        root=args.root,
        output_dir=args.output,
        frame_stride=args.frame_stride,
        max_frames_per_scene=args.max_frames_per_scene,
        camera_id=args.camera_id,
        include_empty_frames=args.include_empty_frames,
        jpeg_quality=args.jpeg_quality,
        target_classes=args.target_classes,
        rare_class_boost=args.rare_class_boost,
        max_images_per_class=args.max_images_per_class,
        min_objects_per_exported_frame=args.min_objects_per_exported_frame,
    )
    print("GT distribution before export [train]:")
    print(
        summarize_class_distribution(
            count_gt_visible_boxes_by_class(
                args.root,
                "train",
                args.train_scenes,
                camera_id=args.camera_id,
                max_frames_per_scene=args.max_frames_per_scene,
                frame_stride=args.frame_stride,
            )
        )
    )
    print("")
    print("GT distribution before export [val]:")
    print(
        summarize_class_distribution(
            count_gt_visible_boxes_by_class(
                args.root,
                "val",
                args.val_scenes,
                camera_id=args.camera_id,
                max_frames_per_scene=args.max_frames_per_scene,
                frame_stride=args.frame_stride,
            )
        )
    )

    train_records = exporter.export_split("train", args.train_scenes)
    val_records = exporter.export_split("val", args.val_scenes)
    records = train_records + val_records
    exporter.write_data_yaml(args.output / "data.yaml")
    exporter.write_manifest(args.output / "export_manifest.csv")
    stats = exporter.class_distribution(records)

    print("")
    print("Wrote data.yaml: %s" % (args.output / "data.yaml"))
    print("Wrote manifest: %s" % (args.output / "export_manifest.csv"))
    print("train images: %d" % len(train_records))
    print("val images: %d" % len(val_records))
    print("total objects: %d" % stats["total_objects"])
    print("class counts after export: %s" % stats["class_counts"])
    print("images per class after export: %s" % stats["image_counts_by_class"])
    print("ignored empty frames: %d" % stats["ignored_empty_frames"])
    print("saved empty frames: %d" % stats["saved_empty_frames"])


def build_arg_parser() -> argparse.ArgumentParser:
    """Build CLI parser."""
    parser = argparse.ArgumentParser(description="Export balanced SmartSpaces YOLO dataset.")
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--train-scenes", nargs="+", required=True)
    parser.add_argument("--val-scenes", nargs="+", required=True)
    parser.add_argument("--camera-id", default=None)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--max-frames-per-scene", type=int, default=None)
    parser.add_argument("--target-classes", nargs="+", default=["Forklift", "PalletTruck"])
    parser.add_argument("--rare-class-boost", type=int, default=3)
    parser.add_argument("--max-images-per-class", type=int, default=None)
    parser.add_argument("--min-objects-per-exported-frame", type=int, default=1)
    parser.add_argument("--include-empty-frames", action="store_true")
    parser.add_argument("--jpeg-quality", type=int, default=95)
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    export_yolo_dataset_balanced(args)


if __name__ == "__main__":
    main()

