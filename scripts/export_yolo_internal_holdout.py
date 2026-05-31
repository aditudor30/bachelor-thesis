"""Export internal YOLO holdout dataset for diagnostics."""

import argparse
from pathlib import Path
from typing import Any, Optional

from deep_oc_sort_3d.detection2d.yolo_allclass_exporter import AllClassYoloExporter


def export_yolo_internal_holdout(args: Any) -> None:
    """Export internal holdout from train scenes."""
    camera_id = _parse_camera_id(args.camera_id)
    exporter = AllClassYoloExporter(
        root=args.root,
        output_dir=args.output,
        train_scenes=[],
        val_scenes=[],
        camera_id=camera_id,
        frame_stride=args.frame_stride,
        max_frames_per_scene=args.max_frames_per_scene,
        include_empty_frames=args.include_empty_frames,
        jpeg_quality=args.jpeg_quality,
    )
    records = exporter.export_internal_holdout(args.holdout_scenes, args.output)
    print("Internal holdout is diagnostic only; it is not an official score.")
    print("Wrote data.yaml: %s" % (args.output / "data.yaml"))
    print("Wrote manifest: %s" % (args.output / "export_manifest.csv"))
    print("images: %d" % len(records))


def _parse_camera_id(value: str) -> Optional[str]:
    if value is None or str(value).lower() == "all":
        return None
    return str(value)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export internal YOLO holdout dataset.")
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--holdout-scenes", nargs="+", required=True)
    parser.add_argument("--camera-id", default="all")
    parser.add_argument("--frame-stride", type=int, default=5)
    parser.add_argument("--max-frames-per-scene", type=int, default=1000)
    parser.add_argument("--include-empty-frames", action="store_true")
    parser.add_argument("--jpeg-quality", type=int, default=95)
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    export_yolo_internal_holdout(args)


if __name__ == "__main__":
    main()

