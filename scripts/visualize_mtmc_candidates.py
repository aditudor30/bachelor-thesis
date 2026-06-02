"""Visualize MTMC candidates with BEV or frame overlays."""

import argparse
from pathlib import Path
from typing import Any, Iterable, List

import cv2

from deep_oc_sort_3d.data.sample_dataset import SmartSpacesFrameDataset
from deep_oc_sort_3d.mtmc.candidate_io import read_candidates_file
from deep_oc_sort_3d.mtmc.candidate_visualization import (
    plot_candidate_3d_trajectories_bev,
    plot_candidate_counts_by_class,
    plot_candidate_lengths_by_class,
    visualize_candidate_on_frame,
)


def visualize_mtmc_candidates(args: Any) -> None:
    """Run requested MTMC candidate visualizations."""
    candidates = read_candidates_file(args.candidates)
    if not args.include_rejected:
        candidates = [item for item in candidates if item.is_candidate]
    if args.bev_output is not None:
        plot_candidate_3d_trajectories_bev(candidates, args.bev_output, max_candidates=args.max_candidates)
        print("Saved BEV: %s" % args.bev_output)
    if args.counts_output is not None:
        plot_candidate_counts_by_class(candidates, args.counts_output)
        print("Saved counts: %s" % args.counts_output)
    if args.lengths_output is not None:
        plot_candidate_lengths_by_class(candidates, args.lengths_output)
        print("Saved lengths: %s" % args.lengths_output)
    if args.output_frame is not None or args.output_dir is not None:
        _visualize_frames(args, candidates)


def _visualize_frames(args: Any, candidates) -> None:
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
    for frame_id in _progress_iter(frame_ids, args.progress and len(frame_ids) > 1, "visualize MTMC candidates"):
        sample = dataset[int(frame_id)]
        image_rgb = sample.get("rgb")
        if image_rgb is None:
            print("warning: missing RGB frame %d" % int(frame_id))
            continue
        drawn = visualize_candidate_on_frame(
            image_rgb,
            candidates,
            int(frame_id),
            include_rejected=args.include_rejected,
        )
        output = _output_path(args, int(frame_id))
        output.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output), cv2.cvtColor(drawn, cv2.COLOR_RGB2BGR))
        print("Saved %s" % output)


def _frame_ids(args: Any) -> List[int]:
    if args.start_frame is not None and args.end_frame is not None:
        return list(range(int(args.start_frame), int(args.end_frame) + 1))
    return [int(args.frame_id)]


def _output_path(args: Any, frame_id: int) -> Path:
    if args.output_dir is not None:
        return args.output_dir / ("%s_%s_f%06d_candidates.png" % (args.scene, args.camera_id, int(frame_id)))
    return args.output_frame


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
    parser = argparse.ArgumentParser(description="Visualize MTMC candidates.")
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("--candidates", required=True, type=Path)
    parser.add_argument("--split", choices=["train", "val", "test"], default="val")
    parser.add_argument("--scene", default="")
    parser.add_argument("--camera-id", default="")
    parser.add_argument("--frame-id", type=int, default=0)
    parser.add_argument("--start-frame", type=int, default=None)
    parser.add_argument("--end-frame", type=int, default=None)
    parser.add_argument("--output-frame", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--bev-output", type=Path, default=None)
    parser.add_argument("--counts-output", type=Path, default=None)
    parser.add_argument("--lengths-output", type=Path, default=None)
    parser.add_argument("--max-candidates", type=int, default=100)
    parser.add_argument("--include-rejected", action="store_true")
    progress_group = parser.add_mutually_exclusive_group()
    progress_group.add_argument("--progress", dest="progress", action="store_true")
    progress_group.add_argument("--no-progress", dest="progress", action="store_false")
    parser.set_defaults(progress=True)
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    visualize_mtmc_candidates(args)


if __name__ == "__main__":
    main()
