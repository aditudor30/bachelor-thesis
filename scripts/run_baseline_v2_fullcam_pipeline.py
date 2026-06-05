"""Run the complete baseline_v2_pseudo3d_fullcam pipeline."""

import argparse
from pathlib import Path

from deep_oc_sort_3d.baseline_v2_fullcam.fullcam_pipeline_runner import run_fullcam_pipeline


def build_arg_parser() -> argparse.ArgumentParser:
    """Build CLI parser."""
    parser = argparse.ArgumentParser(description="Run baseline_v2_pseudo3d_fullcam full pipeline.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    progress_group = parser.add_mutually_exclusive_group()
    progress_group.add_argument("--progress", dest="progress", action="store_true")
    progress_group.add_argument("--no-progress", dest="progress", action="store_false")
    parser.set_defaults(progress=True)
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    run_fullcam_pipeline(
        args.config,
        overwrite=args.overwrite,
        progress=args.progress,
        dry_run=args.dry_run,
        skip_existing=args.skip_existing,
    )


if __name__ == "__main__":
    main()
