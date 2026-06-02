"""Summarize final MVP export outputs."""

import argparse
import json
from pathlib import Path
from typing import Any


def summarize_final_export(args: Any) -> None:
    """Print summaries/eval/validation if present."""
    _print_json(args.export_root / "summaries" / "propagation_summary.json", "propagation")
    _print_json(args.export_root / "summaries" / "export_summary.json", "generic_export")
    _print_json(args.export_root / "validation" / "global_validation_summary.json", "validation")
    _print_json(args.export_root / "eval" / "global_eval.json", "eval")


def _print_json(path: Path, label: str) -> None:
    print("== %s ==" % label)
    if not path.exists():
        print("missing: %s" % path)
        return
    data = json.loads(path.read_text(encoding="utf-8"))
    for key in [
        "files",
        "input_records",
        "output_records",
        "assigned_records",
        "unassigned_records",
        "assignment_ratio",
        "rows_written",
        "num_errors",
        "num_warnings",
        "num_records",
        "unique_global_tracks",
        "global_id_purity_mean",
        "fragmentation_approx",
    ]:
        if key in data:
            print("%s: %s" % (key, data.get(key)))


def build_arg_parser() -> argparse.ArgumentParser:
    """Build CLI parser."""
    parser = argparse.ArgumentParser(description="Summarize final MVP export.")
    parser.add_argument("--export-root", required=True, type=Path)
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    summarize_final_export(args)


if __name__ == "__main__":
    main()
