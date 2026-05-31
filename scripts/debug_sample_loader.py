"""Debug SmartSpacesFrameDataset samples without training or full processing."""

import argparse
from pathlib import Path
from typing import Any

import numpy as np

from deep_oc_sort_3d.data.sample_dataset import SmartSpacesFrameDataset
from deep_oc_sort_3d.utils.logging_utils import print_dict


def _shape_text(value: Any) -> str:
    if value is None:
        return "None"
    return str(value.shape)


def _print_sample(sample: Any) -> None:
    depth = sample.get("depth")
    gt_objects = sample.get("gt_objects")
    calibration = sample.get("calibration")
    print(
        "sample scene=%s scene_id=%s camera=%s frame=%s"
        % (
            sample.get("scene_name"),
            sample.get("scene_id"),
            sample.get("camera_id"),
            sample.get("frame_id"),
        )
    )
    print("  rgb shape: %s" % _shape_text(sample.get("rgb")))
    print("  depth shape: %s" % _shape_text(depth))
    if depth is not None:
        print("  depth dtype: %s" % depth.dtype)
        finite = np.isfinite(depth)
        if finite.any():
            print("  depth min/max: %s / %s" % (float(np.nanmin(depth[finite])), float(np.nanmax(depth[finite]))))
        else:
            print("  depth min/max: unavailable")
    print("  gt_objects: %s" % ("None" if gt_objects is None else len(gt_objects)))
    print("  calibration exists: %s" % (calibration is not None))
    print("  rgb_path: %s" % sample.get("rgb_path"))
    print("  depth_path: %s" % sample.get("depth_path"))
    warnings = sample.get("warnings", [])
    for warning in warnings:
        print("  warning: %s" % warning)


def debug_sample_loader(args: Any) -> None:
    """Create a sample dataset and print the first few samples."""
    dataset = SmartSpacesFrameDataset(
        root=args.root,
        split=args.split,
        scene_name=args.scene,
        max_frames=args.max_frames,
        camera_id=args.camera_id,
        load_rgb=not args.no_rgb,
        load_depth=not args.no_depth,
        load_gt=not args.no_gt,
        depth_dataset_name=args.depth_dataset_name,
    )

    print("Summary:")
    print_dict(dataset.summary(), indent=2)
    print("")
    print("Available cameras: %s" % (", ".join(dataset.list_cameras()) if dataset.list_cameras() else "none"))

    if args.split == "test":
        print("depth is None, as expected for test split")
        print("gt_objects is None, as expected for test split")

    print("")
    limit = min(args.max_frames, len(dataset))
    for idx in range(limit):
        _print_sample(dataset[idx])


def build_arg_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser."""
    parser = argparse.ArgumentParser(description="Debug SmartSpaces frame samples.")
    parser.add_argument("--root", required=True, type=Path, help="Path to MTMC_Tracking_2026.")
    parser.add_argument("--split", required=True, choices=["train", "val", "test"], help="Dataset split.")
    parser.add_argument("--scene", required=True, help="Scene name, for example Warehouse_000.")
    parser.add_argument("--max-frames", type=int, default=3, help="Number of sample frames to inspect.")
    parser.add_argument("--camera-id", default=None, help="Optional camera id, for example Camera_0000.")
    parser.add_argument("--depth-dataset-name", default=None, help="Optional internal HDF5 dataset name.")
    parser.add_argument("--no-rgb", action="store_true", help="Do not read RGB frames.")
    parser.add_argument("--no-depth", action="store_true", help="Do not read depth frames.")
    parser.add_argument("--no-gt", action="store_true", help="Do not load ground truth.")
    return parser


def main() -> None:
    """CLI entrypoint."""
    parser = build_arg_parser()
    args = parser.parse_args()
    debug_sample_loader(args)


if __name__ == "__main__":
    main()
