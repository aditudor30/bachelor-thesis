"""Build standardized Observation3D JSONL from YOLO detections."""

import argparse
from pathlib import Path
from typing import Any, Dict

from deep_oc_sort_3d.observations.observation_builder import Observation3DBuilder
from deep_oc_sort_3d.observations.observation_io import write_observations_jsonl


def build_observations3d(args: Any) -> None:
    """Create observations and write them to JSONL."""
    builder = Observation3DBuilder(
        root=args.root,
        split=args.split,
        scene_name=args.scene,
        yolo_detections_csv=args.detections,
        camera_id=args.camera_id,
        depth_sampling_method=args.depth_sampling_method,
        iou_threshold=args.iou_threshold,
        class_must_match=args.class_must_match,
        use_depth_if_available=not args.no_depth,
    )
    observations = builder.build(max_frames=args.max_frames)
    write_observations_jsonl(observations, args.output)
    print("Wrote observations: %s" % args.output)
    _print_summary(builder.summary(observations))
    if args.split == "test":
        print("test split: ground truth and depth are absent by design; matched_gt should be 0.")


def _print_summary(summary: Dict[str, Any]) -> None:
    print("Summary:")
    for key in [
        "num_detections",
        "num_observations",
        "matched_gt",
        "unmatched",
        "mean_iou",
        "depth_valid",
        "center_3d_available",
    ]:
        print("  %s: %s" % (key, summary.get(key)))
    print("  per_class_counts: %s" % summary.get("per_class_counts", {}))


def build_arg_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser."""
    parser = argparse.ArgumentParser(description="Build Observation3D JSONL from YOLO detections.")
    parser.add_argument("--root", required=True, type=Path, help="Path to MTMC_Tracking_2026.")
    parser.add_argument("--split", required=True, choices=["train", "val", "test"], help="Dataset split.")
    parser.add_argument("--scene", required=True, help="Scene name, for example Warehouse_000.")
    parser.add_argument("--detections", required=True, type=Path, help="YOLO detections CSV.")
    parser.add_argument("--camera-id", default=None, help="Optional camera id filter.")
    parser.add_argument("--output", required=True, type=Path, help="Output Observation3D JSONL path.")
    parser.add_argument("--max-frames", type=int, default=None, help="Optional frame limit.")
    parser.add_argument("--iou-threshold", type=float, default=0.3, help="Detection-GT IoU threshold.")
    parser.add_argument("--depth-sampling-method", default="center_median", help="Depth sampling method.")
    parser.add_argument("--no-depth", action="store_true", help="Disable depth sampling even on train/val.")
    class_group = parser.add_mutually_exclusive_group()
    class_group.add_argument("--class-must-match", dest="class_must_match", action="store_true")
    class_group.add_argument("--no-class-must-match", dest="class_must_match", action="store_false")
    parser.set_defaults(class_must_match=True)
    return parser


def main() -> None:
    """CLI entrypoint."""
    parser = build_arg_parser()
    args = parser.parse_args()
    build_observations3d(args)


if __name__ == "__main__":
    main()

