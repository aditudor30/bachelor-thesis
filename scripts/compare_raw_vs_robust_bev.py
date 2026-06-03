"""Create a side-by-side panel comparing raw and robust BEV plots."""

import argparse
from pathlib import Path

from deep_oc_sort_3d.visualization3d.paper_figure_builder import build_paper_panel_from_images


def main() -> None:
    args = parse_args()
    summary = build_paper_panel_from_images(
        [args.raw_image, args.robust_image],
        args.output,
        titles=["Raw BEV", "Robust BEV"],
        caption_labels=["A", "B"],
    )
    print("images: %d" % int(summary.get("images", 0)))
    print("output: %s" % args.output)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-image", type=Path, required=True)
    parser.add_argument("--robust-image", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--progress", dest="progress", action="store_true", default=True)
    parser.add_argument("--no-progress", dest="progress", action="store_false")
    return parser.parse_args()


if __name__ == "__main__":
    main()

