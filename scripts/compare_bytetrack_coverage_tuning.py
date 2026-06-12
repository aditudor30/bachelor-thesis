"""Recompute Step 21C comparisons without rerunning tracking."""

import argparse
from pathlib import Path

from deep_oc_sort_3d.bytetrack_tuning.tuning_config import load_tuning_config
from deep_oc_sort_3d.bytetrack_tuning.tuning_report import compare_tuning_runs


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare ByteTrack coverage tuning outputs")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--progress", dest="progress", action="store_true")
    group.add_argument("--no-progress", dest="progress", action="store_false")
    parser.set_defaults(progress=True)
    args = parser.parse_args()
    result = compare_tuning_runs(load_tuning_config(args.config), progress=args.progress)
    selection = result.get("selection", {})
    print("selected_variant: %s" % selection.get("selected_variant"))
    print("verdict: %s" % selection.get("verdict", {}).get("label"))


if __name__ == "__main__":
    main()

