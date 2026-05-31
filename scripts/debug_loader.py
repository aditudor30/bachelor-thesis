"""Debug the minimal SmartSpaces dataset loader."""

import argparse
from pathlib import Path
from typing import Any

from deep_oc_sort_3d.data.smartspaces_dataset import SmartSpacesDataset
from deep_oc_sort_3d.utils.logging_utils import print_dict


def _print_paths(title: str, paths: Any, limit: int) -> None:
    print(title)
    if paths is None:
        print("  None")
        return
    if not paths:
        print("  none")
        return
    for path in paths[:limit]:
        print("  %s" % path)


def debug_loader(root: Path, split: str, scene: str, max_frames: int) -> None:
    """Create the minimal loader and print metadata samples."""
    dataset = SmartSpacesDataset(root=root, split=split, scene_name=scene, max_frames=max_frames)

    summary = dataset.summary()
    print("Summary:")
    print_dict(summary, indent=2)

    structure = summary.get("structure", {})
    if structure.get("scene_expected_for_split") is False:
        print("")
        print("Warning:")
        for note in structure.get("notes", []):
            print("  %s" % note)

    calibrations = dataset.load_calibrations()
    print("")
    print("Calibrations: %d camera(s)" % len(calibrations))
    for camera_id, calibration in sorted(calibrations.items())[:5]:
        print(
            "  %s: fps=%s size=%sx%s"
            % (camera_id, calibration.fps, calibration.frame_width, calibration.frame_height)
        )

    ground_truth = dataset.load_ground_truth()
    print("")
    if split == "test":
        print("Ground truth: None (expected for test split)")
        print("Depth maps: None (expected for test split)")
    else:
        if ground_truth is None:
            print("Ground truth: missing or unavailable")
        else:
            print("Ground truth objects loaded: %d" % len(ground_truth))
            for obj in ground_truth[:max_frames]:
                print(
                    "  frame=%d object_id=%d object_type=%s"
                    % (obj.frame_id, obj.object_id, obj.object_type)
                )

    print("")
    _print_paths("Video files:", dataset.list_video_files(), max_frames)
    if split != "test":
        _print_paths("Depth files:", dataset.list_depth_files(), max_frames)


def build_arg_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser."""
    parser = argparse.ArgumentParser(description="Debug SmartSpacesDataset without decoding video.")
    parser.add_argument("--root", required=True, type=Path, help="Path to MTMC_Tracking_2026.")
    parser.add_argument("--split", required=True, choices=["train", "val", "test"], help="Dataset split.")
    parser.add_argument("--scene", required=True, help="Scene name, for example Warehouse_000.")
    parser.add_argument("--max-frames", type=int, default=3, help="Small display limit for files/objects.")
    return parser


def main() -> None:
    """CLI entrypoint."""
    parser = build_arg_parser()
    args = parser.parse_args()
    debug_loader(args.root, args.split, args.scene, args.max_frames)


if __name__ == "__main__":
    main()
