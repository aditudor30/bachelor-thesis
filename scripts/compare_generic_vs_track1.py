"""Compare generic MVP export rows against final Track 1 rows."""

import argparse
import json
from pathlib import Path
from typing import Any

from deep_oc_sort_3d.final_export.track1_dedup_audit import audit_generic_to_track1_dedup


def compare_generic_vs_track1(args: Any) -> None:
    """Compare generic export with Track 1 output and save JSON."""
    report = audit_generic_to_track1_dedup(
        args.generic_export_root,
        args.track1,
        show_progress=args.progress,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print("generic_rows_total: %s" % report.get("generic_rows_total"))
    print("official_rows_total: %s" % report.get("official_rows_total"))
    print("duplicate_rows_removed_estimated: %s" % report.get("duplicate_rows_removed_estimated"))
    print("Wrote %s" % args.output)


def build_arg_parser() -> argparse.ArgumentParser:
    """Build CLI parser."""
    parser = argparse.ArgumentParser(description="Compare generic export and Track 1 output.")
    parser.add_argument("--generic-export-root", required=True, type=Path)
    parser.add_argument("--track1", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    progress_group = parser.add_mutually_exclusive_group()
    progress_group.add_argument("--progress", dest="progress", action="store_true")
    progress_group.add_argument("--no-progress", dest="progress", action="store_false")
    parser.set_defaults(progress=True)
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    compare_generic_vs_track1(args)


if __name__ == "__main__":
    main()
