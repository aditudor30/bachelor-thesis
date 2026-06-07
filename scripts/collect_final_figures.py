"""Collect final freeze figures."""

import argparse
from pathlib import Path

from deep_oc_sort_3d.final_freeze.figure_collector import collect_final_figures_from_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect final freeze figures.")
    parser.add_argument("--config", default="deep_oc_sort_3d/configs/final_freeze.yaml")
    parser.add_argument("--progress", dest="progress", action="store_true")
    parser.add_argument("--no-progress", dest="progress", action="store_false")
    parser.set_defaults(progress=True)
    args = parser.parse_args()
    result = collect_final_figures_from_config(Path(args.config), show_progress=bool(args.progress))
    print("figures:", result.get("num_figures", 0))


if __name__ == "__main__":
    main()

