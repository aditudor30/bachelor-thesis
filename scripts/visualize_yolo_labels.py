"""Visualize exported YOLO labels on a grid of images."""

import argparse
from pathlib import Path
from typing import Any, Dict

import cv2
import yaml

from deep_oc_sort_3d.detection2d.yolo_label_utils import read_yolo_label_file
from deep_oc_sort_3d.detection2d.yolo_visualization import draw_yolo_labels_on_image, make_image_grid


def visualize_yolo_labels(args: Any) -> None:
    """Save a grid of images with YOLO labels drawn."""
    class_names = _load_names(args.dataset / "data.yaml")
    images_dir = args.dataset / "images" / args.split
    labels_dir = args.dataset / "labels" / args.split
    panels = []
    for image_path in sorted(images_dir.glob("*.*"))[: args.max_images]:
        image_bgr = cv2.imread(str(image_path))
        if image_bgr is None:
            continue
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        labels = read_yolo_label_file(labels_dir / (image_path.stem + ".txt"))
        panels.append(draw_yolo_labels_on_image(image_rgb, labels, class_names))
    grid = make_image_grid(panels, cols=4)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.output), cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
    print("Saved %s" % args.output)


def _load_names(path: Path) -> Dict[int, str]:
    if not path.exists():
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    names = data.get("names", {}) if isinstance(data, dict) else {}
    return {int(key): str(value) for key, value in names.items()}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Visualize YOLO labels.")
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--split", required=True, choices=["train", "val"])
    parser.add_argument("--max-images", type=int, default=16)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    visualize_yolo_labels(args)


if __name__ == "__main__":
    main()

