"""Create the final MVP paper/demo panel from selected images."""

import argparse
from pathlib import Path

from deep_oc_sort_3d.visualization3d.paper_figure_builder import build_mvp_demo_panel


def main() -> None:
    args = parse_args()
    summary = build_mvp_demo_panel(args.tracking_image, args.cuboid_image, args.bev_image, args.output)
    print("images: %d" % int(summary.get("images", 0)))
    print("output: %s" % args.output)
    if args.pdf_output is not None:
        build_mvp_demo_panel(args.tracking_image, args.cuboid_image, args.bev_image, args.pdf_output)
        print("pdf_output: %s" % args.pdf_output)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tracking-image", type=Path, required=True)
    parser.add_argument("--cuboid-image", type=Path, required=True)
    parser.add_argument("--bev-image", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pdf-output", type=Path, default=None)
    parser.add_argument("--progress", dest="progress", action="store_true", default=True)
    parser.add_argument("--no-progress", dest="progress", action="store_false")
    return parser.parse_args()


if __name__ == "__main__":
    main()

