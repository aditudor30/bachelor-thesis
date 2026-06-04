"""Summarize Step 15B output artifacts."""

import argparse
from pathlib import Path
from typing import Any

from deep_oc_sort_3d.audit3d.audit3d_io import read_json_if_exists


def run(args: Any) -> None:
    summary = read_json_if_exists(args.root / "summary" / "step15b_summary.json")
    if not summary:
        print("No Step 15B summary found under %s" % args.root)
        return
    print("Classes: %s" % summary.get("class_count"))
    print("Confidence counts: %s" % summary.get("confidence_counts"))
    print("Fallback-required classes: %s" % summary.get("fallback_required_classes"))
    print("Comparison warnings: %s" % summary.get("comparison_warning_counts"))
    print("Next step: %s" % summary.get("next_step"))


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize Step 15B outputs.")
    parser.add_argument("--root", required=True, type=Path)
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
