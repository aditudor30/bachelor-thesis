"""Rebuild summary artifacts for an existing local tracker benchmark."""

import argparse
from pathlib import Path
from typing import List, Optional

from deep_oc_sort_3d.local_tracker_benchmark.benchmark_runner import summarize_benchmark_outputs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize Step 21A local tracker outputs")
    parser.add_argument("--root", type=Path, required=True)
    return parser


def main(argv: Optional[List[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    result = summarize_benchmark_outputs(args.root)
    selected = result.get("selected", {})
    print("trackers summarized: %d" % len(result.get("rows", [])))
    print("selected_tracker: %s" % selected.get("selected_tracker"))
    print("verdict: %s" % selected.get("verdict"))


if __name__ == "__main__":
    main()
