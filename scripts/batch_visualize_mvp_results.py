"""Create a small batch of MVP visualization images."""

import argparse
from pathlib import Path
from typing import List, Optional

import cv2

from deep_oc_sort_3d.data.calibration import load_calibration_json
from deep_oc_sort_3d.data.dataset_structure import get_scene_paths
from deep_oc_sort_3d.data.frame_io import list_video_files, safe_read_video_frame
from deep_oc_sort_3d.visualization3d.frame_visualization import draw_global_frame_records
from deep_oc_sort_3d.visualization3d.visualization_io import filter_records_by_frame, load_global_frame_records_csv


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    written = 0
    for subset in args.subsets:
        subset_root = Path(args.records_root) / subset
        scenes = [path for path in sorted(subset_root.iterdir()) if path.is_dir()] if subset_root.exists() else []
        for scene_dir in scenes[: args.max_scenes]:
            scene_name = scene_dir.name
            split = subset_to_split(subset)
            csv_files = sorted(scene_dir.glob("*_global_records.csv"))[: args.max_cameras]
            for csv_path in csv_files:
                camera_id = csv_path.name.replace("_global_records.csv", "")
                written += visualize_one(args.root, split, scene_name, camera_id, csv_path, args.frames, output_root, args)
    print("images_written: %d" % written)
    print("output_root: %s" % output_root)


def visualize_one(
    root: Path,
    split: str,
    scene_name: str,
    camera_id: str,
    records_path: Path,
    frame_ids: List[int],
    output_root: Path,
    args: argparse.Namespace,
) -> int:
    scene_paths = get_scene_paths(root, split, scene_name)
    video_path = find_video_path(scene_paths.videos_dir, camera_id)
    if video_path is None:
        print("warning: missing video for %s %s" % (scene_name, camera_id))
        return 0
    calibration = None
    if scene_paths.calibration_path is not None and scene_paths.calibration_path.exists():
        calibration = load_calibration_json(scene_paths.calibration_path).get(camera_id)
    records = load_global_frame_records_csv(records_path)
    written = 0
    for frame_id in frame_ids:
        image = safe_read_video_frame(video_path, int(frame_id))
        if image is None:
            print("warning: could not read %s frame %d" % (video_path, int(frame_id)))
            continue
        frame_records = filter_records_by_frame(records, int(frame_id))
        annotated, summary = draw_global_frame_records(
            image,
            frame_records,
            calibration=calibration,
            draw_2d=args.draw_2d,
            draw_3d=args.draw_3d,
            draw_labels=True,
        )
        output = output_root / split / scene_name / ("%s_frame_%06d.png" % (camera_id, int(frame_id)))
        output.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output), cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))
        written += 1
        print(
            "%s records=%d bbox=%d cuboids=%d failed=%d"
            % (
                output,
                len(frame_records),
                int(summary.get("bbox_drawn", 0)),
                int(summary.get("cuboid_projected", 0)),
                int(summary.get("cuboid_failed", 0)),
            )
        )
    return written


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--records-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--subsets", nargs="+", default=["official_val"])
    parser.add_argument("--frames", nargs="+", type=int, default=[100, 200])
    parser.add_argument("--max-scenes", type=int, default=1)
    parser.add_argument("--max-cameras", type=int, default=1)
    parser.add_argument("--progress", dest="progress", action="store_true", default=True)
    parser.add_argument("--no-progress", dest="progress", action="store_false")
    draw_2d = parser.add_mutually_exclusive_group()
    draw_2d.add_argument("--draw-2d", dest="draw_2d", action="store_true", default=True)
    draw_2d.add_argument("--no-draw-2d", dest="draw_2d", action="store_false")
    draw_3d = parser.add_mutually_exclusive_group()
    draw_3d.add_argument("--draw-3d", dest="draw_3d", action="store_true", default=True)
    draw_3d.add_argument("--no-draw-3d", dest="draw_3d", action="store_false")
    return parser.parse_args()


def subset_to_split(subset: str) -> str:
    """Map export subset names back to dataset split names."""
    if subset == "official_val":
        return "val"
    if subset == "test":
        return "test"
    return "train"


def find_video_path(videos_dir: Optional[Path], camera_id: str) -> Optional[Path]:
    if videos_dir is None:
        return None
    for path in list_video_files(videos_dir):
        if path.stem == camera_id:
            return path
    return None


if __name__ == "__main__":
    main()

