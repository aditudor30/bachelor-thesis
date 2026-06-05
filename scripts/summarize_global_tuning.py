"""Print a compact summary for a global tuning root."""

import argparse
from pathlib import Path

from deep_oc_sort_3d.global_tuning.tuning_io import read_json


def build_arg_parser() -> argparse.ArgumentParser:
    """Build CLI parser."""
    parser = argparse.ArgumentParser(description="Summarize global tuning outputs.")
    parser.add_argument("--root", required=True, type=Path)
    return parser


def main() -> None:
    """CLI entrypoint."""
    args = build_arg_parser().parse_args()
    summary = read_json(args.root / "comparison" / "global_tuning_summary.json") or {}
    recommendation = read_json(args.root / "comparison" / "best_run_recommendation.json") or {}
    runs = summary.get("runs", [])
    print("root: %s" % args.root)
    print("runs: %d" % len(runs))
    print("best_run: %s" % recommendation.get("best_run", summary.get("best_run_recommendation", {}).get("best_run")))
    print("verdict: %s" % recommendation.get("verdict", summary.get("best_run_recommendation", {}).get("verdict")))
    for row in runs:
        print(
            "%s frag=%s purity=%s false_merge=%s track1_rows=%s"
            % (
                row.get("run_name"),
                row.get("fragmentation_approx"),
                row.get("global_purity_mean"),
                row.get("false_merge_rate"),
                row.get("track1_rows"),
            )
        )


if __name__ == "__main__":
    main()

