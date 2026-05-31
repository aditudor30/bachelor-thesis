"""Visualize standardized Observation3D records on one RGB frame."""

import argparse
from pathlib import Path
from typing import Any

import cv2

from deep_oc_sort_3d.data.sample_dataset import SmartSpacesFrameDataset
from deep_oc_sort_3d.observations.observation_io import read_observations_jsonl
from deep_oc_sort_3d.observations.observation_visualization import (
    draw_observations_on_image,
    filter_observations,
)


def visualize_observations3d(args: Any) -> None:
    """Draw observations for one frame and save a PNG."""
    dataset = SmartSpacesFrameDataset(
        root=args.root,
        split=args.split,
        scene_name=args.scene,
        max_frames=args.frame_id + 1,
        camera_id=args.camera_id,
        load_rgb=True,
        load_depth=False,
        load_gt=False,
    )
    sample = dataset[args.frame_id]
    image_rgb = sample.get("rgb")
    if image_rgb is None:
        raise IOError("Could not read RGB frame %d for %s" % (args.frame_id, args.camera_id))

    observations = read_observations_jsonl(args.observations)
    frame_observations = filter_observations(observations, args.camera_id, args.frame_id)
    drawn = draw_observations_on_image(image_rgb, frame_observations)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.output), cv2.cvtColor(drawn, cv2.COLOR_RGB2BGR))
    print("observations on frame: %d" % len(frame_observations))
    print("Saved %s" % args.output)


def build_arg_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser."""
    parser = argparse.ArgumentParser(description="Visualize Observation3D JSONL on one RGB frame.")
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--split", required=True, choices=["train", "val", "test"])
    parser.add_argument("--scene", required=True)
    parser.add_argument("--observations", required=True, type=Path)
    parser.add_argument("--camera-id", required=True)
    parser.add_argument("--frame-id", type=int, required=True)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main() -> None:
    """CLI entrypoint."""
    parser = build_arg_parser()
    args = parser.parse_args()
    visualize_observations3d(args)


if __name__ == "__main__":
    main()

