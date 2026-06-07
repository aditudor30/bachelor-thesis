"""Collect final baseline metrics."""

import argparse
from pathlib import Path

from deep_oc_sort_3d.final_freeze.metric_collector import collect_final_metrics_from_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect final freeze metrics.")
    parser.add_argument("--config", default="deep_oc_sort_3d/configs/final_freeze.yaml")
    parser.add_argument("--progress", dest="progress", action="store_true")
    parser.add_argument("--no-progress", dest="progress", action="store_false")
    parser.set_defaults(progress=True)
    args = parser.parse_args()
    result = collect_final_metrics_from_config(Path(args.config), show_progress=bool(args.progress))
    print("baseline rows:", len(result.get("baseline_rows", [])))
    print("track1 rows:", len(result.get("track1", [])))
    print("pseudo3d:", result.get("pseudo3d", {}))
    print("reid:", result.get("reid", {}))


if __name__ == "__main__":
    main()

