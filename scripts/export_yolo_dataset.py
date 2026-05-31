"""Export SmartSpaces train/val GT to a YOLO dataset."""

import argparse
from pathlib import Path
from typing import Any

from deep_oc_sort_3d.detection2d.yolo_dataset_exporter import YoloDatasetExporter


def export_yolo_dataset(args: Any) -> None:
    """Export requested train/val scenes."""
    exporter = YoloDatasetExporter(
        root=args.root,
        output_dir=args.output,
        frame_stride=args.frame_stride,
        max_frames_per_scene=args.max_frames_per_scene,
        camera_id=args.camera_id,
        include_empty_frames=args.include_empty_frames,
        jpeg_quality=args.jpeg_quality,
    )
    train_records = exporter.export_split("train", args.train_scenes)
    val_records = exporter.export_split("val", args.val_scenes)
    records = train_records + val_records
    data_yaml = args.output / "data.yaml"
    exporter.write_data_yaml(data_yaml)
    stats = exporter.class_distribution(records)

    print("Wrote data.yaml: %s" % data_yaml)
    print("train images: %d" % len(train_records))
    print("val images: %d" % len(val_records))
    print("labels: %d" % len(records))
    print("total objects: %d" % stats["total_objects"])
    print("class counts: %s" % stats["class_counts"])
    print("ignored empty frames: %d" % stats["ignored_empty_frames"])
    print("saved empty frames: %d" % stats["saved_empty_frames"])


def build_arg_parser() -> argparse.ArgumentParser:
    """Build CLI parser."""
    parser = argparse.ArgumentParser(description="Export SmartSpaces GT to YOLO format.")
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--train-scenes", nargs="+", required=True)
    parser.add_argument("--val-scenes", nargs="+", required=True)
    parser.add_argument("--camera-id", default=None)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--max-frames-per-scene", type=int, default=None)
    parser.add_argument("--include-empty-frames", action="store_true")
    parser.add_argument("--jpeg-quality", type=int, default=95)
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    export_yolo_dataset(args)


if __name__ == "__main__":
    main()

