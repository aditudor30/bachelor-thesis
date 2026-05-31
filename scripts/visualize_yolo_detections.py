"""Visualize YOLO detections on one SmartSpaces frame."""

import argparse
from pathlib import Path
from typing import Any

import cv2

from deep_oc_sort_3d.data.dataset_structure import get_scene_paths
from deep_oc_sort_3d.data.frame_io import infer_camera_id_from_video_path, list_video_files, safe_read_video_frame
from deep_oc_sort_3d.detection2d.yolo_detection_io import read_detections_csv
from deep_oc_sort_3d.detection2d.yolo_visualization import draw_detections_on_image


def visualize_yolo_detections(args: Any) -> None:
    """Draw detections for one frame."""
    scene_paths = get_scene_paths(args.root, args.split, args.scene)
    video_path = None
    for path in list_video_files(scene_paths.videos_dir):
        if infer_camera_id_from_video_path(path) == args.camera_id:
            video_path = path
            break
    if video_path is None:
        raise IOError("Could not find video for camera %s" % args.camera_id)
    frame_rgb = safe_read_video_frame(video_path, args.frame_id)
    if frame_rgb is None:
        raise IOError("Could not read frame %d" % args.frame_id)
    detections = [
        det
        for det in read_detections_csv(args.detections)
        if det.camera_id == args.camera_id and det.frame_id == args.frame_id
    ]
    drawn = draw_detections_on_image(frame_rgb, detections)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.output), cv2.cvtColor(drawn, cv2.COLOR_RGB2BGR))
    print("Saved %s" % args.output)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Visualize YOLO detections.")
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--split", required=True, choices=["train", "val", "test"])
    parser.add_argument("--scene", required=True)
    parser.add_argument("--detections", required=True, type=Path)
    parser.add_argument("--camera-id", required=True)
    parser.add_argument("--frame-id", type=int, required=True)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    visualize_yolo_detections(args)


if __name__ == "__main__":
    main()

