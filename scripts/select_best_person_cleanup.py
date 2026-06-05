"""CLI for selecting best Person cleanup run."""

import argparse
from pathlib import Path

from deep_oc_sort_3d.person_cleanup.person_cleanup_io import read_json, write_json
from deep_oc_sort_3d.person_cleanup.person_cleanup_selection import select_best_person_cleanup


def build_arg_parser() -> argparse.ArgumentParser:
    """Build CLI parser."""
    parser = argparse.ArgumentParser(description="Select best Person cleanup run.")
    parser.add_argument("--comparison", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    summary = read_json(args.comparison) or {}
    recommendation = summary.get("best_person_cleanup_recommendation")
    if not isinstance(recommendation, dict):
        recommendation = select_best_person_cleanup(summary.get("runs", []), summary.get("v2_current", {}), {})
    write_json(recommendation, args.output)
    print("verdict: %s" % recommendation.get("verdict"))
    print("best_run: %s" % recommendation.get("best_run"))


if __name__ == "__main__":
    main()

