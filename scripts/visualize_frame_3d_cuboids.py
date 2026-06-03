"""Visualize one RGB frame with 2D boxes and projected 3D cuboids."""

import argparse
from pathlib import Path
from typing import Any, List, Optional

import cv2

from deep_oc_sort_3d.data.calibration import load_calibration_json
from deep_oc_sort_3d.data.dataset_structure import get_scene_paths
from deep_oc_sort_3d.data.frame_io import list_video_files, safe_read_video_frame
from deep_oc_sort_3d.visualization3d.frame_visualization import draw_global_frame_records
from deep_oc_sort_3d.visualization3d.visualization_io import (
    filter_records_by_class,
    filter_records_by_frame,
    filter_records_by_global_track_id,
    load_global_frame_records_csv,
)


def main() -> None:
    args = parse_args()
    records = load_global_frame_records_csv(args.records)
    records = filter_records_by_frame(records, args.frame_id)
    records = [record for record in records if str(record.get("camera_id", "")) == args.camera_id]
    if args.class_name is not None:
        records = filter_records_by_class(records, args.class_name)
    if args.global_track_id is not None:
        records = filter_records_by_global_track_id(records, args.global_track_id)

    scene_paths = get_scene_paths(args.root, args.split, args.scene)
    video_path = find_video_path(scene_paths.videos_dir, args.camera_id)
    if video_path is None:
        raise FileNotFoundError("Missing video for %s %s" % (args.scene, args.camera_id))
    image = safe_read_video_frame(video_path, args.frame_id)
    if image is None:
        raise IOError("Could not read frame %d from %s" % (args.frame_id, video_path))

    calibration = None
    if scene_paths.calibration_path is not None and scene_paths.calibration_path.exists():
        calibration = load_calibration_json(scene_paths.calibration_path).get(args.camera_id)

    annotated, summary = draw_global_frame_records(
        image,
        records,
        calibration=calibration,
        draw_2d=args.draw_2d,
        draw_3d=args.draw_3d,
        draw_labels=True,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output), cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))
    print("records: %d" % len(records))
    print("bbox_drawn: %d" % int(summary.get("bbox_drawn", 0)))
    print("cuboid_projected: %d" % int(summary.get("cuboid_projected", 0)))
    print("cuboid_failed: %d" % int(summary.get("cuboid_failed", 0)))
    print("output: %s" % output)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--camera-id", required=True)
    parser.add_argument("--frame-id", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--class-name", default=None)
    parser.add_argument("--global-track-id", type=int, default=None)
    parser.add_argument("--progress", dest="progress", action="store_true", default=True)
    parser.add_argument("--no-progress", dest="progress", action="store_false")
    draw_2d = parser.add_mutually_exclusive_group()
    draw_2d.add_argument("--draw-2d", dest="draw_2d", action="store_true", default=True)
    draw_2d.add_argument("--no-draw-2d", dest="draw_2d", action="store_false")
    draw_3d = parser.add_mutually_exclusive_group()
    draw_3d.add_argument("--draw-3d", dest="draw_3d", action="store_true", default=True)
    draw_3d.add_argument("--no-draw-3d", dest="draw_3d", action="store_false")
    return parser.parse_args()


def find_video_path(videos_dir: Optional[Path], camera_id: str) -> Optional[Path]:
    """Find the video path for one camera."""
    if videos_dir is None:
        return None
    for path in list_video_files(videos_dir):
        if path.stem == camera_id:
            return path
    return None


if __name__ == "__main__":
    main()

