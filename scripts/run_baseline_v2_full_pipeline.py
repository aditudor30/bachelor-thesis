"""Run or print the full baseline_v2 pseudo-3D pipeline."""

import argparse
from pathlib import Path

from deep_oc_sort_3d.pseudo3d_integration.baseline_v2_runner import run_baseline_v2_pipeline


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run baseline_v2 pseudo-3D full pipeline.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    progress_group = parser.add_mutually_exclusive_group()
    progress_group.add_argument("--progress", dest="progress", action="store_true")
    progress_group.add_argument("--no-progress", dest="progress", action="store_false")
    parser.set_defaults(progress=True)
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    run_baseline_v2_pipeline(args.config, overwrite=args.overwrite, progress=args.progress, dry_run=args.dry_run)


if __name__ == "__main__":
    main()

