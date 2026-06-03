"""Validate Track 1 submission or unconfirmed preview."""

import argparse
from pathlib import Path
from typing import Any

from deep_oc_sort_3d.final_export.track1_export_types import (
    default_unconfirmed_track1_schema,
    load_track1_schema_yaml,
)
from deep_oc_sort_3d.final_export.track1_validator import (
    print_track1_validation_report,
    validate_track1_export,
    write_track1_validation_report,
)


def validate_submission(args: Any) -> None:
    """Validate Track 1 output."""
    schema = load_track1_schema_yaml(args.schema_yaml) if args.schema_yaml else default_unconfirmed_track1_schema()
    report = validate_track1_export(args.submission, schema, show_progress=args.progress)
    write_track1_validation_report(report, args.output)
    print_track1_validation_report(report)
    print("Wrote %s" % args.output)
    if args.fail_on_errors and int(report.get("num_errors", 0)) > 0:
        raise SystemExit(1)


def build_arg_parser() -> argparse.ArgumentParser:
    """Build CLI parser."""
    parser = argparse.ArgumentParser(description="Validate Track 1 submission or preview.")
    parser.add_argument("--submission", required=True, type=Path)
    parser.add_argument("--schema-yaml", type=Path, default=None)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--fail-on-errors", action="store_true")
    progress_group = parser.add_mutually_exclusive_group()
    progress_group.add_argument("--progress", dest="progress", action="store_true")
    progress_group.add_argument("--no-progress", dest="progress", action="store_false")
    parser.set_defaults(progress=True)
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    validate_submission(args)


if __name__ == "__main__":
    main()
