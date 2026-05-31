"""Inspect depth value statistics for a small number of frames."""

import argparse
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from deep_oc_sort_3d.data.sample_dataset import SmartSpacesFrameDataset
from deep_oc_sort_3d.geometry.depth_quality import guess_depth_unit, summarize_depth_array


def _print_stats(stats: Dict[str, Any]) -> None:
    print("  shape: %s" % (stats.get("shape"),))
    print("  dtype: %s" % stats.get("dtype"))
    print("  min/max: %s / %s" % (stats.get("min"), stats.get("max")))
    print("  mean: %s" % stats.get("mean"))
    print("  median: %s" % stats.get("median"))
    print("  valid_ratio: %.6f" % float(stats.get("valid_ratio", 0.0)))
    print("  num_zero: %s" % stats.get("num_zero"))
    print("  percentiles: %s" % stats.get("percentiles"))
    print("  unit_guess: %s" % guess_depth_unit(stats))


def inspect_depth_statistics(args: Any) -> None:
    """Print depth stats for the first max_frames frames."""
    dataset = SmartSpacesFrameDataset(
        root=args.root,
        split=args.split,
        scene_name=args.scene,
        max_frames=args.max_frames,
        camera_id=args.camera_id,
        load_rgb=False,
        load_depth=True,
        load_gt=False,
        depth_dataset_name=args.depth_dataset_name,
    )

    medians = []
    unit_guesses = []
    for idx in range(min(args.max_frames, len(dataset))):
        sample = dataset[idx]
        depth = sample.get("depth")
        print("")
        print("frame_id: %d" % sample.get("frame_id"))
        if depth is None:
            print("  depth: None")
            continue
        stats = summarize_depth_array(depth, sample_name="frame_%d" % sample.get("frame_id"))
        _print_stats(stats)
        if stats.get("median") is not None:
            medians.append(float(stats["median"]))
        unit_guesses.append(guess_depth_unit(stats))

    print("")
    print("Summary:")
    if medians:
        aggregate_stats = summarize_depth_array(np.asarray(medians, dtype=float), sample_name="frame_medians")
        recommendation = guess_depth_unit(aggregate_stats)
        print("  frame_median_count: %d" % len(medians))
        print("  median_of_frame_medians: %s" % aggregate_stats.get("median"))
        print("  mean_of_frame_medians: %s" % aggregate_stats.get("mean"))
        print("  unit_guesses: %s" % ", ".join(unit_guesses))
        print("  recommended_unit: %s" % recommendation)
    else:
        print("  no valid depth frames inspected")


def build_arg_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser."""
    parser = argparse.ArgumentParser(description="Inspect depth statistics lazily.")
    parser.add_argument("--root", required=True, type=Path, help="Path to MTMC_Tracking_2026.")
    parser.add_argument("--split", required=True, choices=["train", "val", "test"], help="Dataset split.")
    parser.add_argument("--scene", required=True, help="Scene name.")
    parser.add_argument("--camera-id", required=True, help="Camera id.")
    parser.add_argument("--max-frames", type=int, default=5, help="Number of frames to inspect.")
    parser.add_argument("--depth-dataset-name", default=None, help="Optional internal HDF5 dataset name.")
    return parser


def main() -> None:
    """CLI entrypoint."""
    parser = build_arg_parser()
    args = parser.parse_args()
    inspect_depth_statistics(args)


if __name__ == "__main__":
    main()

