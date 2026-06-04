"""Run local tracking for baseline_v2 pseudo-3D observations."""

import argparse
from pathlib import Path

from deep_oc_sort_3d.scripts.run_batch_local_tracking import run_batch_local_tracking


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run baseline_v2 local tracking.")
    parser.add_argument("--config", type=Path, default=Path("deep_oc_sort_3d/configs/baseline_v2_local_tracking.yaml"))
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
            "run_root",
            "output_root",
            "subsets",
            "scenes",
            "camera_ids",
            "mode",
            "min_confidence",
            "min_hits",
            "max_misses",
            "cost_threshold",
            "max_3d_distance",
            "min_iou",
            "class_must_match",
            "max_detections_per_frame",
        ],
    )
    run_batch_local_tracking(args)


def _ensure_attrs(args, names) -> None:
    for name in names:
        if not hasattr(args, name):
            setattr(args, name, None)


if __name__ == "__main__":
    main()
