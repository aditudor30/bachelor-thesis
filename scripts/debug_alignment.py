"""Debug RGB-depth-GT-calibration alignment for a few frames."""

import argparse
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from deep_oc_sort_3d.data.sample_dataset import SmartSpacesFrameDataset
from deep_oc_sort_3d.training.target_builder import TrainingTargetBuilder


def _array_text(value: Any) -> str:
    if value is None:
        return "None"
    return np.array2string(np.asarray(value, dtype=float), precision=3, suppress_small=True)


def debug_alignment(args: Any) -> None:
    """Print numeric alignment diagnostics for a small frame window."""
    dataset = SmartSpacesFrameDataset(
        root=args.root,
        split=args.split,
        scene_name=args.scene,
        max_frames=args.max_frames,
        camera_id=args.camera_id,
        load_rgb=False,
        load_depth=True,
        load_gt=True,
        depth_dataset_name=args.depth_dataset_name,
    )
    builder = TrainingTargetBuilder()

    if args.split == "test":
        print("Test split has no ground truth and no depth maps. Alignment targets are not available.")

    total_targets = 0
    total_visible = 0
    total_depth_valid = 0
    errors = []
    frame_limit = min(args.max_frames, len(dataset))

    for idx in range(frame_limit):
        sample = dataset[idx]
        frame_targets = builder.build_targets_from_sample(sample)
        gt_objects = sample.get("gt_objects")
        gt_count = 0 if gt_objects is None else len(gt_objects)
        visible_targets = [target for target in frame_targets.targets if target.bbox_xyxy is not None]
        bbox_camera_counts = _bbox_camera_counts(gt_objects)

        print("")
        print("frame_id: %d" % frame_targets.frame_id)
        print("gt_objects: %d" % gt_count)
        print("visible_targets_for_camera: %d" % len(visible_targets))
        if len(visible_targets) == 0 and bbox_camera_counts:
            print("bbox cameras in this frame: %s" % _format_camera_counts(bbox_camera_counts))

        for target in visible_targets:
            print(
                "  object_id=%d class=%s bbox=%s"
                % (target.object_id, target.class_name, target.bbox_xyxy)
            )
            print("    center_3d: %s" % _array_text(target.center_3d))
            print("    dimensions_3d: %s" % _array_text(target.dimensions_3d))
            print("    yaw: %.6f" % target.yaw)
            print("    depth_value: %s" % target.depth_value)
            print("    backprojection_error: %s" % target.backprojection_error)

        total_targets += len(frame_targets.targets)
        total_visible += len(visible_targets)
        for target in frame_targets.targets:
            if target.depth_value is not None:
                total_depth_valid += 1
            if target.backprojection_error is not None:
                errors.append(float(target.backprojection_error))

    missing_bbox = total_targets - total_visible
    missing_bbox_pct = 0.0
    if total_targets > 0:
        missing_bbox_pct = float(missing_bbox) / float(total_targets) * 100.0

    print("")
    print("Summary:")
    print("  total_targets: %d" % total_targets)
    print("  targets_with_visible_bbox: %d" % total_visible)
    print("  targets_with_valid_depth: %d" % total_depth_valid)
    if errors:
        print("  mean_backprojection_error: %.6f" % float(np.mean(errors)))
        print("  median_backprojection_error: %.6f" % float(np.median(errors)))
    else:
        print("  mean_backprojection_error: None")
        print("  median_backprojection_error: None")
    print("  targets_without_bbox_pct: %.2f" % missing_bbox_pct)


def _bbox_camera_counts(gt_objects: Any) -> Dict[str, int]:
    counts = {}
    if gt_objects is None:
        return counts
    for obj in gt_objects:
        for camera_id in obj.visible_bboxes_2d.keys():
            counts[camera_id] = counts.get(camera_id, 0) + 1
    return counts


def _format_camera_counts(counts: Dict[str, int]) -> str:
    parts = []
    for camera_id in sorted(counts.keys()):
        parts.append("%s=%d" % (camera_id, counts[camera_id]))
    return ", ".join(parts)


def build_arg_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser."""
    parser = argparse.ArgumentParser(description="Debug RGB-depth-GT-calibration alignment.")
    parser.add_argument("--root", required=True, type=Path, help="Path to MTMC_Tracking_2026.")
    parser.add_argument("--split", required=True, choices=["train", "val", "test"], help="Dataset split.")
    parser.add_argument("--scene", required=True, help="Scene name, for example Warehouse_000.")
    parser.add_argument("--camera-id", required=True, help="Camera id, for example Camera_0000.")
    parser.add_argument("--max-frames", type=int, default=5, help="Small frame count to inspect.")
    parser.add_argument("--depth-dataset-name", default=None, help="Optional internal HDF5 dataset name.")
    return parser


def main() -> None:
    """CLI entrypoint."""
    parser = build_arg_parser()
    args = parser.parse_args()
    debug_alignment(args)


if __name__ == "__main__":
    main()
