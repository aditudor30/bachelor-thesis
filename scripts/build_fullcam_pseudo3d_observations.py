"""Build baseline_v2_pseudo3d_fullcam Observation3D files."""

import argparse
from pathlib import Path
from typing import Any, Dict

from deep_oc_sort_3d.baseline_v2_fullcam.fullcam_observation_builder import (
    build_fullcam_observations_from_config,
    load_fullcam_config,
)


def run(args: Any) -> Dict[str, Any]:
    """Run the fullcam observation builder."""
    config = load_fullcam_config(args.config)
    summary = build_fullcam_observations_from_config(
        config,
        show_progress=bool(args.progress),
        overwrite=bool(args.overwrite),
        subsets=args.subsets,
        scenes=args.scenes,
        camera_ids=args.camera_ids,
    )
    print("files: %s" % summary.get("files"))
    print("output_observations: %s" % summary.get("output_observations"))
    print("pseudo3d_used_rate: %s" % summary.get("pseudo3d_used_rate"))
    print("fallback_original_used_rate: %s" % summary.get("fallback_original_used_rate"))
    print("fullcam_valid_for_pipeline: %s" % summary.get("fullcam_valid_for_pipeline"))
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    """Build CLI parser."""
    parser = argparse.ArgumentParser(description="Build fullcam stabilized pseudo3D Observation3D files.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--subsets", nargs="+", default=None)
    parser.add_argument("--scenes", nargs="+", default=None)
    parser.add_argument("--camera-ids", nargs="+", default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    progress_group = parser.add_mutually_exclusive_group()
    progress_group.add_argument("--progress", dest="progress", action="store_true")
    progress_group.add_argument("--no-progress", dest="progress", action="store_false")
    parser.set_defaults(progress=True)
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    if args.skip_existing:
        args.overwrite = False
    run(args)


if __name__ == "__main__":
    main()
