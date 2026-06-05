"""Compare completed global tuning runs."""

import argparse
from pathlib import Path

from deep_oc_sort_3d.global_tuning.tuning_comparison import compare_global_tuning_runs
from deep_oc_sort_3d.global_tuning.tuning_config import load_sweep_config
from deep_oc_sort_3d.global_tuning.tuning_report import write_global_tuning_report


def build_arg_parser() -> argparse.ArgumentParser:
    """Build CLI parser."""
    parser = argparse.ArgumentParser(description="Compare global tuning runs.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--overwrite", action="store_true")
    progress_group = parser.add_mutually_exclusive_group()
    progress_group.add_argument("--progress", dest="progress", action="store_true")
    progress_group.add_argument("--no-progress", dest="progress", action="store_false")
    parser.set_defaults(progress=True)
    return parser


def main() -> None:
    """CLI entrypoint."""
    args = build_arg_parser().parse_args()
    comparison = compare_global_tuning_runs(args.config, progress=args.progress)
    sweep = load_sweep_config(args.config)
    output_root = Path(sweep.get("paths", {}).get("output_root", "output/global_tuning/debug"))
    report_path = output_root / "comparison" / "GLOBAL_ASSOCIATION_TUNING_REPORT.md"
    write_global_tuning_report(comparison, report_path)
    print("runs: %d" % len(comparison.get("runs", [])))
    print("best_run: %s" % comparison.get("best_run_recommendation", {}).get("best_run"))
    print("report: %s" % report_path)


if __name__ == "__main__":
    main()
