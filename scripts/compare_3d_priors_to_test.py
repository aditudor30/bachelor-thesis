"""Compare final 3D priors with generic export dimensions."""

import argparse
from pathlib import Path
from typing import Any, Dict

from deep_oc_sort_3d.audit3d.audit3d_io import read_json_if_exists, write_csv, write_json
from deep_oc_sort_3d.audit3d.generic_3d_audit import read_generic_export_rows
from deep_oc_sort_3d.priors3d.dimension_prior_analysis import (
    build_dimension_comparison_report,
    compare_priors_to_generic_rows,
    comparison_to_rows,
)
from deep_oc_sort_3d.audit3d.audit3d_io import write_markdown


def run(args: Any) -> Dict[str, Any]:
    """Run prior-vs-generic comparison."""
    priors = read_json_if_exists(args.priors)
    generic_rows = read_generic_export_rows(args.generic_export_root, show_progress=args.progress)
    config = {
        "compare_subsets": args.compare_subsets,
        "dimension_ratio_warning_low": args.dimension_ratio_warning_low,
        "dimension_ratio_warning_high": args.dimension_ratio_warning_high,
    }
    summary = compare_priors_to_generic_rows(priors, generic_rows, config)
    args.output_root.mkdir(parents=True, exist_ok=True)
    write_json(summary, args.output_root / "dimension_prior_vs_test_comparison.json")
    write_csv(comparison_to_rows(summary), args.output_root / "dimension_prior_vs_test_comparison.csv")
    write_markdown(build_dimension_comparison_report(summary), args.output_root / "dimension_prior_vs_test_comparison.md")
    print("Comparison rows: %s" % summary.get("row_count"))
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare class 3D priors with generic export dimensions.")
    parser.add_argument("--priors", required=True, type=Path)
    parser.add_argument("--generic-export-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--compare-subsets", nargs="*", default=["official_val", "internal_holdout", "test"])
    parser.add_argument("--dimension-ratio-warning-low", type=float, default=0.5)
    parser.add_argument("--dimension-ratio-warning-high", type=float, default=2.0)
    progress_group = parser.add_mutually_exclusive_group()
    progress_group.add_argument("--progress", dest="progress", action="store_true")
    progress_group.add_argument("--no-progress", dest="progress", action="store_false")
    parser.set_defaults(progress=True)
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()

