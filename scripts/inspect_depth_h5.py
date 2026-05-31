"""Inspect large HDF5 depth files without loading them fully."""

import argparse
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from deep_oc_sort_3d.data.depth_io import (
    inspect_h5_depth_file,
    list_depth_files,
    safe_read_depth_frame_h5,
)


def _print_report(report: Dict[str, Any], max_candidates: int) -> None:
    print("path: %s" % report.get("path"))
    print("exists: %s" % report.get("exists"))
    print("selected_dataset: %s" % report.get("selected_dataset"))
    print("selected_layout: %s" % report.get("selected_layout"))
    print("shape: %s" % (report.get("shape"),))
    print("dtype: %s" % report.get("dtype"))
    print("num_frames: %s" % report.get("num_frames"))
    if report.get("error"):
        print("error: %s" % report.get("error"))
    print("dataset_candidates:")
    candidates = report.get("dataset_candidates", [])
    if not candidates:
        print("  none")
    for candidate in candidates[:max_candidates]:
        print(
            "  %s shape=%s dtype=%s num_frames=%s"
            % (
                candidate.get("name"),
                candidate.get("shape"),
                candidate.get("dtype"),
                candidate.get("num_frames"),
            )
        )
    if len(candidates) > max_candidates:
        print("  ... %d more candidate(s)" % (len(candidates) - max_candidates))


def _print_frame_stats(path: Path, frame_idx: int, dataset_name: Any) -> None:
    frame = safe_read_depth_frame_h5(path, frame_idx, dataset_name)
    if frame is None:
        print("frame %d: could not be read" % frame_idx)
        return
    print("frame %d shape: %s" % (frame_idx, frame.shape))
    print("frame %d dtype: %s" % (frame_idx, frame.dtype))
    finite = np.isfinite(frame)
    if finite.any():
        print("frame %d min: %s" % (frame_idx, float(np.nanmin(frame[finite]))))
        print("frame %d max: %s" % (frame_idx, float(np.nanmax(frame[finite]))))
        print("frame %d mean: %s" % (frame_idx, float(np.nanmean(frame[finite]))))
    else:
        print("frame %d stats: unavailable" % frame_idx)


def inspect_paths(paths: List[Path], read_frame: Any, dataset_name: Any, max_candidates: int) -> None:
    """Inspect one or more HDF5 paths."""
    for index, path in enumerate(paths):
        if index > 0:
            print("")
        report = inspect_h5_depth_file(path)
        _print_report(report, max_candidates)
        if read_frame is not None:
            _print_frame_stats(path, int(read_frame), dataset_name)


def build_arg_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser."""
    parser = argparse.ArgumentParser(description="Inspect SmartSpaces HDF5 depth files lazily.")
    parser.add_argument("--path", type=Path, default=None, help="Path to one Camera_XXXX.h5 file.")
    parser.add_argument("--scene-root", type=Path, default=None, help="Path to one scene directory.")
    parser.add_argument("--max-files", type=int, default=2, help="Max files to inspect from --scene-root.")
    parser.add_argument("--max-candidates", type=int, default=20, help="Max internal datasets to print per file.")
    parser.add_argument("--read-frame", type=int, default=None, help="Optionally read one frame index.")
    parser.add_argument("--dataset-name", default=None, help="Optional internal HDF5 dataset name.")
    return parser


def main() -> None:
    """CLI entrypoint."""
    parser = build_arg_parser()
    args = parser.parse_args()

    paths = []
    if args.path is not None:
        paths = [args.path]
    elif args.scene_root is not None:
        paths = list_depth_files(args.scene_root / "depth_maps")[: args.max_files]
    else:
        parser.error("Provide either --path or --scene-root.")

    if not paths:
        print("No HDF5 depth files found.")
        return
    inspect_paths(paths, args.read_frame, args.dataset_name, args.max_candidates)


if __name__ == "__main__":
    main()
