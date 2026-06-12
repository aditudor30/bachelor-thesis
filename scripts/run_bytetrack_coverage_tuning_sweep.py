"""Run the two-phase Step 21C ByteTrack coverage sweep."""

import argparse
from pathlib import Path

from deep_oc_sort_3d.bytetrack_tuning.tuning_config import load_tuning_config
from deep_oc_sort_3d.bytetrack_tuning.tuning_report import compare_tuning_runs
from deep_oc_sort_3d.bytetrack_tuning.tuning_sweep_runner import run_tuning_sweep


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run coverage-oriented ByteTrack tuning sweep")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--variant", default=None)
    parser.add_argument("--full-all-variants", action="store_true")
    parser.add_argument("--top-k", type=int, default=None)
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
    result = run_tuning_sweep(
        config=config,
        variant=args.variant,
        full_all_variants=args.full_all_variants,
        top_k=args.top_k,
        progress=args.progress,
        overwrite=args.overwrite,
        skip_existing=args.skip_existing,
        dry_run=args.dry_run,
    )
    print("selected_for_full: %s" % ", ".join(result.get("selected_for_full", [])))
    if not args.dry_run:
        comparison = compare_tuning_runs(config, progress=args.progress)
        print("verdict: %s" % comparison.get("selection", {}).get("verdict", {}).get("label"))


if __name__ == "__main__":
    main()

