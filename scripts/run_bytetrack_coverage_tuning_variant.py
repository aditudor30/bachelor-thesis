"""Run one complete Step 21C ByteTrack tuning variant."""

import argparse
from pathlib import Path

from deep_oc_sort_3d.bytetrack_tuning.tuning_config import load_tuning_config
from deep_oc_sort_3d.bytetrack_tuning.tuning_report import compare_tuning_runs
from deep_oc_sort_3d.bytetrack_tuning.tuning_stage_runner import run_tuning_variant


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run one ByteTrack coverage tuning variant")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--local-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--progress", dest="progress", action="store_true")
    group.add_argument("--no-progress", dest="progress", action="store_false")
    parser.set_defaults(progress=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_tuning_config(args.config)
    result = run_tuning_variant(
        config=config,
        variant_name=args.variant,
        full_pipeline=not args.local_only,
        progress=args.progress,
        overwrite=args.overwrite,
        skip_existing=args.skip_existing,
        dry_run=args.dry_run,
    )
    print("variant: %s" % result.get("variant"))
    print("status: %s" % result.get("status"))
    if not args.dry_run:
        compare_tuning_runs(config, progress=args.progress)


if __name__ == "__main__":
    main()

