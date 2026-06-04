"""Build local tracklets for baseline_v2 pseudo-3D."""

import argparse
from pathlib import Path

from deep_oc_sort_3d.scripts.build_batch_local_tracklets import build_batch_local_tracklets


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build baseline_v2 tracklets.")
    parser.add_argument("--config", type=Path, default=Path("deep_oc_sort_3d/configs/baseline_v2_tracklets.yaml"))
    parser.add_argument("--overwrite", action="store_true")
    progress_group = parser.add_mutually_exclusive_group()
    progress_group.add_argument("--progress", dest="progress", action="store_true", default=None)
    progress_group.add_argument("--no-progress", dest="progress", action="store_false")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    _ensure_attrs(
        args,
        [
            "tracking_root",
            "output_root",
            "subsets",
            "scenes",
            "camera_ids",
            "min_length",
            "min_mean_confidence",
            "smoothing_window",
            "smooth_trajectory",
        ],
    )
    build_batch_local_tracklets(args)


def _ensure_attrs(args, names) -> None:
    for name in names:
        if not hasattr(args, name):
            setattr(args, name, None)


if __name__ == "__main__":
    main()
