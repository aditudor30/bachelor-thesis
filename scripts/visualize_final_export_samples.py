"""Visualize final frame-level global export samples."""

import argparse
from pathlib import Path
from typing import Any, Iterable, List

from deep_oc_sort_3d.final_export.export_visualization import visualize_global_export_frame


def visualize_final_export_samples(args: Any) -> None:
    """Visualize one frame or a frame range."""
    if args.start_frame is not None and args.end_frame is not None:
        if args.output_dir is None:
            raise ValueError("--output-dir is required for frame range visualization")
        frame_ids = list(range(int(args.start_frame), int(args.end_frame) + 1))
        for frame_id in _progress_iter(frame_ids, args.progress, "visualize final frames", "frame"):
            output_path = args.output_dir / ("%s_%s_f%06d.png" % (args.scene, args.camera_id, frame_id))
            visualize_global_export_frame(
                args.root,
                args.records,
                args.split,
                args.scene,
                args.camera_id,
                frame_id,
                output_path,
            )
        print("frames: %d" % len(frame_ids))
    else:
        visualize_global_export_frame(
            args.root,
            args.records,
            args.split,
            args.scene,
            args.camera_id,
            args.frame_id,
            args.output,
        )
        print("output: %s" % args.output)


def _progress_iter(values: List[Any], show_progress: bool, desc: str, unit: str) -> Iterable[Any]:
    if not show_progress:
        return values
    try:
        from tqdm import tqdm
    except ImportError:
        return _print_progress_iter(values, desc)
    return tqdm(values, desc=desc, unit=unit)


def _print_progress_iter(values: List[Any], desc: str) -> Iterable[Any]:
    total = len(values)
    for index, value in enumerate(values):
        print("%s: %d/%d frame=%s" % (desc, index + 1, total, value))
        yield value


def build_arg_parser() -> argparse.ArgumentParser:
    """Build CLI parser."""
    parser = argparse.ArgumentParser(description="Visualize final MVP export samples.")
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--records", required=True, type=Path)
    parser.add_argument("--split", required=True)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--camera-id", required=True)
    parser.add_argument("--frame-id", type=int, default=0)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--start-frame", type=int, default=None)
    parser.add_argument("--end-frame", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    progress_group = parser.add_mutually_exclusive_group()
    progress_group.add_argument("--progress", dest="progress", action="store_true")
    progress_group.add_argument("--no-progress", dest="progress", action="store_false")
    parser.set_defaults(progress=True)
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    if args.output is None and args.output_dir is None:
        raise ValueError("Provide --output for one frame or --output-dir for a frame range.")
    visualize_final_export_samples(args)


if __name__ == "__main__":
    main()
