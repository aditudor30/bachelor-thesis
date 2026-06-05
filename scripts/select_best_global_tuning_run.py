"""Select the best global tuning run from a comparison JSON."""

import argparse
from pathlib import Path

from deep_oc_sort_3d.global_tuning.tuning_io import read_json, write_json
from deep_oc_sort_3d.global_tuning.tuning_selection import select_best_run


def build_arg_parser() -> argparse.ArgumentParser:
    """Build CLI parser."""
    parser = argparse.ArgumentParser(description="Select best global tuning run.")
    parser.add_argument("--comparison", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main() -> None:
    """CLI entrypoint."""
    args = build_arg_parser().parse_args()
    comparison = read_json(args.comparison) or {}
    recommendation = comparison.get("best_run_recommendation")
    if not isinstance(recommendation, dict):
        recommendation = select_best_run(
            comparison.get("runs", []),
            comparison.get("v2_current", {}),
            {},
        )
    write_json(recommendation, args.output)
    print("verdict: %s" % recommendation.get("verdict"))
    print("best_run: %s" % recommendation.get("best_run"))


if __name__ == "__main__":
    main()
