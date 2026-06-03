"""Create a grid panel from existing visualization images."""

import argparse
from pathlib import Path
from typing import List, Optional

from deep_oc_sort_3d.visualization3d.figure_panels import make_frame_grid_panel


def main() -> None:
    args = parse_args()
    titles = args.titles if args.titles else None
    summary = make_frame_grid_panel(args.images, args.output, titles=titles)
    print("images: %d" % int(summary.get("images", 0)))
    print("grid: %dx%d" % (int(summary.get("rows", 0)), int(summary.get("cols", 0))))
    print("output: %s" % args.output)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", nargs="+", type=Path, required=True)
    parser.add_argument("--titles", nargs="*", default=None)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    main()

