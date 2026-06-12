"""Compare completed Step 21E variants and regenerate reports."""

import argparse
from pathlib import Path

from deep_oc_sort_3d.bytetrack_motion_filtering.motion_filter_config import load_motion_filter_config
from deep_oc_sort_3d.bytetrack_motion_filtering.motion_filter_report import compare_and_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare Step 21E motion-filter variants")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--variant", default=None, help="Accepted for CLI symmetry; comparison uses completed outputs.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--progress", dest="progress", action="store_true")
    group.add_argument("--no-progress", dest="progress", action="store_false")
    parser.set_defaults(progress=True)
    args = parser.parse_args()
    config = load_motion_filter_config(args.config)
    result = compare_and_report(config, progress=args.progress)
    print("selected_variant: %s" % result.get("selection", {}).get("selected_variant"))
    print("verdict: %s" % result.get("selection", {}).get("verdict"))
    print("report: %s" % result.get("report"))


if __name__ == "__main__":
    main()

