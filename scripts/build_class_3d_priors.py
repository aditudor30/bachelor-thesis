"""Build final class-wise 3D priors from Step 15A prior summaries."""

import argparse
from pathlib import Path
from typing import Any, Dict

from deep_oc_sort_3d.priors3d.class_priors_builder import (
    build_class_priors_report,
    build_final_class_priors,
    final_priors_to_rows,
)
from deep_oc_sort_3d.priors3d.priors_io import read_prior_csv, read_prior_json, write_prior_outputs


def run(args: Any) -> Dict[str, Any]:
    """Run final class prior construction."""
    prior_csv_rows = read_prior_csv(args.input, show_progress=args.progress)
    prior_json = read_prior_json(args.input_json) if args.input_json is not None else {}
    config = {
        "robust_method": args.robust_method,
        "min_count_high_confidence": args.min_count_high_confidence,
        "min_count_medium_confidence": args.min_count_medium_confidence,
        "constant_cv_threshold": args.constant_cv_threshold,
    }
    summary = build_final_class_priors(prior_json=prior_json, prior_csv_rows=prior_csv_rows, config=config)
    rows = final_priors_to_rows(summary)
    report = build_class_priors_report(summary)
    report_path = args.output_report or args.output_json.parent / "class_dimension_priors_report.md"
    write_prior_outputs(summary, rows, args.output_json, args.output_csv, report, report_path)
    print("Final class priors: %s classes" % summary.get("class_count"))
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build final class-wise 3D priors.")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--input-json", type=Path, default=None)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--output-csv", required=True, type=Path)
    parser.add_argument("--output-report", type=Path, default=None)
    parser.add_argument("--robust-method", default="median", choices=["median", "trimmed_mean"])
    parser.add_argument("--min-count-high-confidence", type=int, default=1000)
    parser.add_argument("--min-count-medium-confidence", type=int, default=100)
    parser.add_argument("--constant-cv-threshold", type=float, default=0.02)
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

