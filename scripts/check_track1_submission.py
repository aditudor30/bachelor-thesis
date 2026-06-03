"""Run final sanity checks for Track 1 submission."""

import argparse
import json
from pathlib import Path
from typing import Any, Dict

from deep_oc_sort_3d.final_export.track1_dedup_audit import (
    audit_generic_to_track1_dedup,
    write_dedup_audit_report,
)
from deep_oc_sort_3d.final_export.track1_final_checks import (
    compute_track1_distribution,
    print_track1_final_report,
    read_track1_txt,
    validate_track1_rows,
    write_track1_final_report,
)
from deep_oc_sort_3d.final_export.track1_submission_summary import (
    build_submission_summary,
    write_submission_summary_json,
    write_submission_summary_markdown,
)


def check_track1_submission(args: Any) -> None:
    """Run final checks and write reports."""
    rows = read_track1_txt(args.track1)
    validation_report = validate_track1_rows(
        rows,
        expected_scene_ids=args.expected_scene_ids,
        valid_class_ids=args.valid_class_ids,
        show_progress=args.progress,
    )
    validation_report["track1_path"] = str(args.track1)
    distribution_report = compute_track1_distribution(rows)
    args.output_root.mkdir(parents=True, exist_ok=True)
    write_track1_final_report(validation_report, args.output_root / "final_validation_report.json")
    write_track1_final_report(distribution_report, args.output_root / "final_distribution_report.json")
    dedup_report = {}
    if args.generic_export_root is not None and args.generic_export_root.exists():
        dedup_report = audit_generic_to_track1_dedup(
            args.generic_export_root,
            args.track1,
            show_progress=args.progress,
        )
        write_dedup_audit_report(dedup_report, args.output_root / "dedup_audit_report.json")
    summary = build_submission_summary(
        validation_report,
        dedup_report,
        config_paths=args.config_paths,
        baseline_name=args.baseline_name,
    )
    write_submission_summary_json(summary, args.output_root / "submission_summary.json")
    write_submission_summary_markdown(summary, args.output_root / "final_check_summary.md")
    write_submission_summary_markdown(summary, args.output_root / "submission_summary.md")
    print_track1_final_report(validation_report)
    if dedup_report:
        print("duplicate_rows_removed_estimated: %s" % dedup_report.get("duplicate_rows_removed_estimated"))
    print("Wrote reports to %s" % args.output_root)
    if args.fail_on_errors and int(validation_report.get("num_errors", 0)) > 0:
        raise SystemExit(1)


def build_arg_parser() -> argparse.ArgumentParser:
    """Build CLI parser."""
    parser = argparse.ArgumentParser(description="Run final Track 1 submission checks.")
    parser.add_argument("--track1", required=True, type=Path)
    parser.add_argument("--generic-export-root", type=Path, default=None)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--expected-scene-ids", nargs="+", type=int, default=[23, 24, 25])
    parser.add_argument("--valid-class-ids", nargs="+", type=int, default=[0, 1, 2, 3, 4, 5, 6])
    parser.add_argument("--baseline-name", default="baseline_v1_geometry_only")
    parser.add_argument("--config", dest="config_paths", action="append", type=Path, default=[])
    parser.add_argument("--fail-on-errors", action="store_true")
    progress_group = parser.add_mutually_exclusive_group()
    progress_group.add_argument("--progress", dest="progress", action="store_true")
    progress_group.add_argument("--no-progress", dest="progress", action="store_false")
    parser.set_defaults(progress=True)
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    check_track1_submission(args)


if __name__ == "__main__":
    main()
