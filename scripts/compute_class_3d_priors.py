"""CLI for class-wise 3D dimension priors."""

import argparse
from pathlib import Path
from typing import Any, Dict

from deep_oc_sort_3d.audit3d.audit3d_io import write_csv, write_json
from deep_oc_sort_3d.audit3d.class_3d_priors import (
    class_priors_to_rows,
    compare_class_priors_between_subsets,
    comparison_to_rows,
    compute_class_dimension_priors,
    split_rows_by_subset,
)
from deep_oc_sort_3d.audit3d.generic_3d_audit import read_generic_export_rows


def run(args: Any) -> Dict[str, Any]:
    rows = read_generic_export_rows(args.generic_export_root, show_progress=args.progress)
    prior_rows = split_rows_by_subset(rows, args.use_subsets) if args.use_subsets else rows
    priors = compute_class_dimension_priors(prior_rows)
    test_rows = split_rows_by_subset(rows, ["test"])
    test_priors = compute_class_dimension_priors(test_rows)
    comparison = compare_class_priors_between_subsets(priors, test_priors)
    args.output_root.mkdir(parents=True, exist_ok=True)
    write_json(priors, args.output_root / "class_dimension_priors.json")
    write_csv(class_priors_to_rows(priors), args.output_root / "class_dimension_priors.csv")
    write_csv(comparison_to_rows(comparison), args.output_root / "class_dimension_comparison_test_vs_val.csv")
    print("Class priors: %s classes" % priors.get("class_count"))
    print("Wrote class priors to %s" % args.output_root)
    return priors


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compute class-wise 3D dimension priors.")
    parser.add_argument("--generic-export-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--use-subsets", nargs="*", default=["official_val", "internal_holdout"])
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
