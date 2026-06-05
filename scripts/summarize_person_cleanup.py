"""Print compact Person cleanup summary."""

import argparse
from pathlib import Path

from deep_oc_sort_3d.person_cleanup.person_cleanup_io import read_json


def build_arg_parser() -> argparse.ArgumentParser:
    """Build CLI parser."""
    parser = argparse.ArgumentParser(description="Summarize Person cleanup outputs.")
    parser.add_argument("--root", required=True, type=Path)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    summary = read_json(args.root / "comparison" / "person_cleanup_summary.json") or {}
    recommendation = read_json(args.root / "comparison" / "best_person_cleanup_recommendation.json") or {}
    print("root: %s" % args.root)
    print("runs: %d" % len(summary.get("runs", [])))
    print("best_run: %s" % recommendation.get("best_run", summary.get("best_person_cleanup_recommendation", {}).get("best_run")))
    print("verdict: %s" % recommendation.get("verdict", summary.get("best_person_cleanup_recommendation", {}).get("verdict")))
    for row in summary.get("runs", []):
        print(
            "%s track1_rows=%s person_rows=%s non_person_delta=%s track1_errors=%s"
            % (
                row.get("run_name"),
                row.get("track1_rows"),
                row.get("person_rows"),
                row.get("vs_v2_non_person_rows_delta"),
                row.get("track1_validation_errors"),
            )
        )


if __name__ == "__main__":
    main()

