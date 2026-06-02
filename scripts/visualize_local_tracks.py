"""Visualize local tracks on RGB frames."""

import argparse
from pathlib import Path
from typing import Any, Iterable, List

import cv2

from deep_oc_sort_3d.data.sample_dataset import SmartSpacesFrameDataset
from deep_oc_sort_3d.tracking.track_io import read_local_tracks_csv
from deep_oc_sort_3d.tracking.track_visualization import draw_tracks_on_frame


def visualize_local_tracks(args: Any) -> None:
    """Visualize one frame or a range of frames."""
    records = read_local_tracks_csv(args.tracks)
    frame_ids = _frame_ids(args)
    dataset = SmartSpacesFrameDataset(
        root=args.root,
        split=args.split,
        scene_name=args.scene,
        max_frames=max(frame_ids) + 1,
        camera_id=args.camera_id,
        load_rgb=True,
        load_depth=False,
        load_gt=False,
    )
    for frame_id in _progress_iter(frame_ids, not args.no_progress and len(frame_ids) > 1, "visualize local tracks"):
        sample = dataset[frame_id]
        image_rgb = sample.get("rgb")
        if image_rgb is None:
            print("warning: missing RGB frame %d" % frame_id)
            continue
        frame_records = [record for record in records if record.frame_id == int(frame_id)]
        drawn = draw_tracks_on_frame(image_rgb, frame_records, show_gt=not args.no_gt)
        output = _output_path(args, frame_id)
        output.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output), cv2.cvtColor(drawn, cv2.COLOR_RGB2BGR))
        print("Saved %s with %d tracks" % (output, len(frame_records)))


def _frame_ids(args: Any) -> List[int]:
    if args.start_frame is not None and args.end_frame is not None:
        return list(range(int(args.start_frame), int(args.end_frame) + 1))
    return [int(args.frame_id)]


def _output_path(args: Any, frame_id: int) -> Path:
    if args.output_dir is not None:
        return args.output_dir / ("%s_%s_f%06d.png" % (args.scene, args.camera_id, int(frame_id)))
    return args.output


def _progress_iter(values: List[int], show_progress: bool, desc: str) -> Iterable[int]:
    if not show_progress:
        return values
    try:
        from tqdm import tqdm
    except ImportError:
        return _print_progress_iter(values, desc)
    return tqdm(values, desc=desc, unit="frame")


def _print_progress_iter(values: List[int], desc: str) -> Iterable[int]:
    total = len(values)
    for index, value in enumerate(values):
        print("%s: frame %d/%d" % (desc, index + 1, total))
        yield value


def build_arg_parser() -> argparse.ArgumentParser:
    """Build CLI parser."""
    parser = argparse.ArgumentParser(description="Visualize local tracks.")
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--tracks", required=True, type=Path)
    parser.add_argument("--split", required=True, choices=["train", "val", "test"])
    parser.add_argument("--scene", required=True)
    parser.add_argument("--camera-id", required=True)
    parser.add_argument("--frame-id", type=int, default=0)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--start-frame", type=int, default=None)
    parser.add_argument("--end-frame", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--no-gt", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    if args.output is None and args.output_dir is None:
        raise ValueError("Provide --output for one frame or --output-dir for frame ranges.")
    visualize_local_tracks(args)


if __name__ == "__main__":
    main()
