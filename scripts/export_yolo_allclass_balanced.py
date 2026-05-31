"""Export all-class balanced YOLO train/val dataset."""

import argparse
from pathlib import Path
from typing import Any, Optional

from deep_oc_sort_3d.detection2d.yolo_allclass_exporter import AllClassYoloExporter


def export_yolo_allclass_balanced(args: Any) -> None:
    """Export all-class train/val YOLO dataset."""
    camera_id = _parse_camera_id(args.camera_id)
    exporter = AllClassYoloExporter(
        root=args.root,
        output_dir=args.output,
        train_scenes=args.train_scenes,
        val_scenes=args.val_scenes,
        camera_id=camera_id,
        frame_stride=args.frame_stride,
        max_frames_per_scene=args.max_frames_per_scene,
        target_classes=args.target_classes,
        rare_class_boost=args.rare_class_boost,
        include_empty_frames=args.include_empty_frames,
        jpeg_quality=args.jpeg_quality,
    )
    records = exporter.export_train_val()
    stats = exporter.summarize_export()
    print("Wrote data.yaml: %s" % (args.output / "data.yaml"))
    print("Wrote manifest: %s" % (args.output / "export_manifest.csv"))
    print("images: %d" % len(records))
    print("total objects: %d" % stats.get("total_objects", 0))
    print("class counts: %s" % stats.get("class_counts", {}))
    print("images per class: %s" % stats.get("image_counts_by_class", {}))
    print("ignored empty frames: %d" % stats.get("ignored_empty_frames", 0))
    print("saved empty frames: %d" % stats.get("saved_empty_frames", 0))


def _parse_camera_id(value: str) -> Optional[str]:
    if value is None or str(value).lower() == "all":
        return None
    return str(value)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export all-class balanced YOLO dataset.")
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--train-scenes", nargs="+", required=True)
    parser.add_argument("--val-scenes", nargs="+", required=True)
    parser.add_argument("--camera-id", default="all")
    parser.add_argument("--frame-stride", type=int, default=5)
    parser.add_argument("--max-frames-per-scene", type=int, default=1000)
    parser.add_argument("--target-classes", nargs="+", default=None)
    parser.add_argument("--rare-class-boost", type=int, default=4)
    parser.add_argument("--include-empty-frames", action="store_true")
    parser.add_argument("--jpeg-quality", type=int, default=95)
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    export_yolo_allclass_balanced(args)


if __name__ == "__main__":
    main()

