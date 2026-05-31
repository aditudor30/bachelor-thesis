"""Run YOLO detection on SmartSpaces scene videos and export CSV."""

import argparse
from pathlib import Path
from typing import Any, Dict, List

from deep_oc_sort_3d.data.dataset_structure import get_scene_paths, scene_name_to_id
from deep_oc_sort_3d.data.frame_io import infer_camera_id_from_video_path, list_video_files
from deep_oc_sort_3d.detection2d.yolo_detection_io import write_detections_csv, write_mot_like_detections
from deep_oc_sort_3d.detection2d.yolo_inference import load_yolo_model, run_yolo_on_video


def run_yolo_inference(args: Any) -> None:
    """Run YOLO over one scene."""
    model = load_yolo_model(args.model)
    scene_paths = get_scene_paths(args.root, args.split, args.scene)
    videos = list_video_files(scene_paths.videos_dir)
    if args.camera_id is not None:
        videos = [path for path in videos if infer_camera_id_from_video_path(path) == args.camera_id]
    scene_id = scene_name_to_id(args.scene)
    if scene_id is None:
        scene_id = -1

    all_detections = []
    by_camera = {}
    for video_path in videos:
        camera_id = infer_camera_id_from_video_path(video_path)
        detections = run_yolo_on_video(
            model=model,
            video_path=video_path,
            scene_id=scene_id,
            scene_name=args.scene,
            split=args.split,
            camera_id=camera_id,
            conf_threshold=args.conf,
            max_frames=args.max_frames,
            frame_stride=args.frame_stride,
            imgsz=args.imgsz,
        )
        all_detections.extend(detections)
        by_camera[camera_id] = detections

    write_detections_csv(all_detections, args.output)
    print("Wrote %d detections to %s" % (len(all_detections), args.output))
    if args.mot_output_dir is not None:
        args.mot_output_dir.mkdir(parents=True, exist_ok=True)
        for camera_id, detections in by_camera.items():
            path = args.mot_output_dir / ("%s_%s.txt" % (args.scene, camera_id))
            write_mot_like_detections(detections, path)
            print("Wrote MOT-like detections: %s" % path)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run YOLO inference on SmartSpaces videos.")
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--split", required=True, choices=["train", "val", "test"])
    parser.add_argument("--scene", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--camera-id", default=None)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--mot-output-dir", type=Path, default=None)
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    run_yolo_inference(args)


if __name__ == "__main__":
    main()

