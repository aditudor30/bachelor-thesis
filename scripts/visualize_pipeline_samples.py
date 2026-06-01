"""Visualize detections and Observation3D records for one pipeline frame."""

import argparse
from pathlib import Path
from typing import Any

import cv2

from deep_oc_sort_3d.data.sample_dataset import SmartSpacesFrameDataset
from deep_oc_sort_3d.detection2d.yolo_detection_io import read_detections_csv
from deep_oc_sort_3d.detection2d.yolo_visualization import draw_detections_on_image
from deep_oc_sort_3d.observations.observation_io import read_observations_jsonl
from deep_oc_sort_3d.observations.observation_visualization import draw_observations_on_image, filter_observations
from deep_oc_sort_3d.pipeline.pipeline_paths import get_detection_csv_path, get_observation_jsonl_path
from deep_oc_sort_3d.pipeline.run_config import DEFAULT_SPLIT_BY_SUBSET, load_pipeline_config


def visualize_pipeline_samples(args: Any) -> None:
    """Save a PNG with detections and observations drawn for one frame."""
    split = _resolve_split(args)
    dataset = SmartSpacesFrameDataset(
        root=args.root,
        split=split,
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

    detections_csv = get_detection_csv_path(args.run_root, args.subset, args.scene, args.camera_id)
    observations_jsonl = get_observation_jsonl_path(args.run_root, args.subset, args.scene, args.camera_id)
    detections = [
        det
        for det in read_detections_csv(detections_csv)
        if det.frame_id == int(args.frame_id) and det.camera_id == args.camera_id
    ]
    observations = filter_observations(read_observations_jsonl(observations_jsonl), args.camera_id, args.frame_id)
    drawn = draw_detections_on_image(image_rgb, detections)
    drawn = draw_observations_on_image(drawn, observations)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.output), cv2.cvtColor(drawn, cv2.COLOR_RGB2BGR))
    print("detections on frame: %d" % len(detections))
    print("observations on frame: %d" % len(observations))
    print("Saved %s" % args.output)


def _resolve_split(args: Any) -> str:
    config_path = args.run_root / "summaries" / "run_config_resolved.yaml"
    if config_path.exists():
        try:
            config = load_pipeline_config(config_path)
            if args.subset in config.split_by_subset:
                return config.split_by_subset[args.subset]
        except Exception as exc:
            print("warning: could not read resolved config for split lookup: %s" % exc)
    return DEFAULT_SPLIT_BY_SUBSET.get(args.subset, args.subset)


def build_arg_parser() -> argparse.ArgumentParser:
    """Build CLI parser."""
    parser = argparse.ArgumentParser(description="Visualize one frame from a pipeline run.")
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--subset", required=True)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--camera-id", required=True)
    parser.add_argument("--frame-id", required=True, type=int)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    visualize_pipeline_samples(args)


if __name__ == "__main__":
    main()
