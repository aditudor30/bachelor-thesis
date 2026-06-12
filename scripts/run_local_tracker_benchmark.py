"""Run the isolated Step 21A local tracker benchmark."""

import argparse
from pathlib import Path
from typing import List, Optional

from deep_oc_sort_3d.local_tracker_benchmark.benchmark_runner import run_local_tracker_benchmark


def build_parser() -> argparse.ArgumentParser:
    """Build CLI parser."""
    parser = argparse.ArgumentParser(description="Benchmark local trackers over existing YOLO11m detections")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--subset",
        choices=("quick", "val", "internal_holdout", "test", "all_available"),
        default="quick",
    )
    parser.add_argument(
        "--trackers",
        nargs="+",
        choices=("current", "bytetrack", "botsort_no_reid", "botsort_sbs_mot17", "botsort_sbs_mot20", "botsort_osnet"),
        default=None,
    )
    progress_group = parser.add_mutually_exclusive_group()
    progress_group.add_argument("--progress", dest="progress", action="store_true")
    progress_group.add_argument("--no-progress", dest="progress", action="store_false")
    parser.set_defaults(progress=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    return parser


def main(argv: Optional[List[str]] = None) -> None:
    """Run benchmark and print its isolated output root."""
    args = build_parser().parse_args(argv)
    result = run_local_tracker_benchmark(
        config_path=args.config,
        subset_name=args.subset,
        requested_trackers=args.trackers,
        progress=bool(args.progress),
        overwrite=bool(args.overwrite),
        skip_existing=bool(args.skip_existing),
    )
    print("Output root: %s" % result.get("output_root"))
    comparison = result.get("comparison", {})
    selected = comparison.get("selected", {}) if isinstance(comparison, dict) else {}
    print("selected_tracker: %s" % selected.get("selected_tracker"))
    print("verdict: %s" % selected.get("verdict"))


if __name__ == "__main__":
    main()
