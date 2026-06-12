"""Run one Step 21E motion-filter variant and its downstream pipeline."""

import argparse
from pathlib import Path

from deep_oc_sort_3d.bytetrack_motion_filtering.motion_filter_config import load_motion_filter_config, write_resolved_config
from deep_oc_sort_3d.bytetrack_motion_filtering.motion_filter_report import compare_and_report
from deep_oc_sort_3d.bytetrack_motion_filtering.motion_filter_sweep_runner import run_motion_filter_sweep


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one ByteTrack gap-aware motion-filter variant")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--progress", dest="progress", action="store_true")
    group.add_argument("--no-progress", dest="progress", action="store_false")
    parser.set_defaults(progress=True)
    args = parser.parse_args()
    config = load_motion_filter_config(args.config)
    write_resolved_config(config)
    run_motion_filter_sweep(
        config,
        variant=args.variant,
        progress=args.progress,
        overwrite=args.overwrite,
        skip_existing=args.skip_existing,
    )
    result = compare_and_report(config, progress=args.progress)
    print("variant: %s" % args.variant)
    print("selected_variant: %s" % result.get("selection", {}).get("selected_variant"))
    print("verdict: %s" % result.get("selection", {}).get("verdict"))


if __name__ == "__main__":
    main()

