"""Visualize local tracklets on RGB frames or BEV plots."""

import argparse
from pathlib import Path
from typing import Any, Iterable, List

import cv2

from deep_oc_sort_3d.data.sample_dataset import SmartSpacesFrameDataset
from deep_oc_sort_3d.tracklets.tracklet_io import read_tracklets_file
from deep_oc_sort_3d.tracklets.tracklet_types import LocalTracklet
from deep_oc_sort_3d.tracklets.tracklet_visualization import (
    plot_tracklet_bev_trajectory,
    visualize_tracklets_on_frame,
)


def visualize_local_tracklets(args: Any) -> None:
    """Run requested local tracklet visualization."""
    tracklets = read_tracklets_file(args.tracklets)
    if args.bev_output is not None:
        tracklet = _find_tracklet(tracklets, args.tracklet_id)
        plot_tracklet_bev_trajectory(tracklet, args.bev_output)
        print("Saved BEV: %s" % args.bev_output)
    if args.output is not None or args.output_dir is not None:
        _visualize_frames(args, tracklets)


def _visualize_frames(args: Any, tracklets: List[LocalTracklet]) -> None:
    if args.root is None:
        raise ValueError("Frame visualization requires --root.")
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
    for frame_id in _progress_iter(frame_ids, args.progress and len(frame_ids) > 1, "visualize local tracklets"):
        sample = dataset[int(frame_id)]
        image_rgb = sample.get("rgb")
        if image_rgb is None:
            print("warning: missing RGB frame %d" % int(frame_id))
            continue
        drawn = visualize_tracklets_on_frame(image_rgb, tracklets, int(frame_id))
        output = _output_path(args, int(frame_id))
        output.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output), cv2.cvtColor(drawn, cv2.COLOR_RGB2BGR))
        print("Saved %s" % output)


def _find_tracklet(tracklets: List[LocalTracklet], tracklet_id: int) -> LocalTracklet:
    for tracklet in tracklets:
        if int(tracklet.local_track_id) == int(tracklet_id):
            return tracklet
    raise ValueError("Tracklet id not found: %s" % tracklet_id)


def _frame_ids(args: Any) -> List[int]:
    if args.start_frame is not None and args.end_frame is not None:
        return list(range(int(args.start_frame), int(args.end_frame) + 1))
    return [int(args.frame_id)]


def _output_path(args: Any, frame_id: int) -> Path:
    if args.output_dir is not None:
        return args.output_dir / ("%s_%s_f%06d_tracklets.png" % (args.scene, args.camera_id, int(frame_id)))
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
    parser = argparse.ArgumentParser(description="Visualize local tracklets.")
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("--tracklets", required=True, type=Path)
    parser.add_argument("--split", choices=["train", "val", "test"], default="val")
    parser.add_argument("--scene", default="")
    parser.add_argument("--camera-id", default="")
    parser.add_argument("--frame-id", type=int, default=0)
    parser.add_argument("--start-frame", type=int, default=None)
    parser.add_argument("--end-frame", type=int, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--tracklet-id", type=int, default=0)
    parser.add_argument("--bev-output", type=Path, default=None)
    progress_group = parser.add_mutually_exclusive_group()
    progress_group.add_argument("--progress", dest="progress", action="store_true")
    progress_group.add_argument("--no-progress", dest="progress", action="store_false")
    parser.set_defaults(progress=True)
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    if args.output is None and args.output_dir is None and args.bev_output is None:
        raise ValueError("Provide --output/--output-dir for frames or --bev-output for BEV.")
    visualize_local_tracklets(args)


if __name__ == "__main__":
    main()
