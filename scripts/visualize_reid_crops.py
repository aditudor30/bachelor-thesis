"""Create a grid image from saved ReID crop debug images."""

import argparse
from pathlib import Path
from typing import List

from deep_oc_sort_3d.visualization3d.figure_panels import make_frame_grid_panel


def main() -> None:
    args = parse_args()
    images = find_crop_images(args.crop_root, args.max_crops)
    if not images:
        raise ValueError("No crop images found under %s" % args.crop_root)
    summary = make_frame_grid_panel(images, args.output)
    print("crops: %d" % len(images))
    print("grid: %sx%s" % (summary.get("rows"), summary.get("cols")))
    print("output: %s" % args.output)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--crop-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-crops", type=int, default=64)
    parser.add_argument("--progress", dest="progress", action="store_true", default=True)
    parser.add_argument("--no-progress", dest="progress", action="store_false")
    return parser.parse_args()


def find_crop_images(root: Path, max_crops: int) -> List[Path]:
    images = []
    for suffix in ("*.jpg", "*.png", "*.jpeg"):
        images.extend(sorted(root.rglob(suffix)))
    return images[: int(max_crops)]


if __name__ == "__main__":
    main()

