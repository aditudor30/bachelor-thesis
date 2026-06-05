"""Package baseline_v2_pseudo3d_fullcam Track1 submission."""

import argparse
from pathlib import Path

from deep_oc_sort_3d.baseline_v2_fullcam.fullcam_observation_builder import load_fullcam_config
from deep_oc_sort_3d.baseline_v2_fullcam.fullcam_package import package_fullcam_submission


def build_arg_parser() -> argparse.ArgumentParser:
    """Build CLI parser."""
    parser = argparse.ArgumentParser(description="Package baseline_v2_pseudo3d_fullcam submission.")
    parser.add_argument("--config", required=True, type=Path)
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
    config = load_fullcam_config(args.config)
    summary = package_fullcam_submission(config, overwrite=args.overwrite, show_progress=args.progress)
    print("package_root: %s" % summary.get("package_root"))
    print("manifest_path: %s" % summary.get("manifest_path"))


if __name__ == "__main__":
    main()
