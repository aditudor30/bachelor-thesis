"""Visualize YOLO samples that contain a requested class."""

import argparse
from pathlib import Path
from typing import Any, Dict, Optional

import cv2
import yaml

from deep_oc_sort_3d.detection2d.yolo_label_utils import read_yolo_label_file
from deep_oc_sort_3d.detection2d.yolo_visualization import draw_yolo_labels_on_image, make_image_grid


def visualize_yolo_class_samples(args: Any) -> None:
    """Save a grid of images that contain a target class."""
    names = _load_names(args.dataset / "data.yaml")
    class_id = _resolve_class_id(args.class_id, args.class_name, names)
    if class_id is None:
        raise ValueError("Could not resolve requested class.")
    images_dir = args.dataset / "images" / args.split
    labels_dir = args.dataset / "labels" / args.split
    panels = []
    for image_path in sorted(images_dir.glob("*.*")):
        label_path = labels_dir / (image_path.stem + ".txt")
        labels = read_yolo_label_file(label_path)
        if not any(label.class_id == class_id for label in labels):
            continue
        image_bgr = cv2.imread(str(image_path))
        if image_bgr is None:
            continue
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        panels.append(draw_yolo_labels_on_image(image_rgb, labels, names))
        if len(panels) >= args.max_images:
            break
    if not panels:
        print("No images found for class_id=%d class_name=%s" % (class_id, names.get(class_id)))
        return
    grid = make_image_grid(panels, cols=4)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.output), cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
    print("Saved %s with %d panels" % (args.output, len(panels)))


def _load_names(path: Path) -> Dict[int, str]:
    if not path.exists():
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    names_raw = data.get("names", {}) if isinstance(data, dict) else {}
    names = {}
    if isinstance(names_raw, dict):
        for key, value in names_raw.items():
            names[int(key)] = str(value)
    elif isinstance(names_raw, list):
        for idx, value in enumerate(names_raw):
            names[int(idx)] = str(value)
    return names


def _resolve_class_id(class_id: Optional[int], class_name: Optional[str], names: Dict[int, str]) -> Optional[int]:
    if class_id is not None:
        return int(class_id)
    if class_name is None:
        return None
    lower_name = str(class_name).lower()
    for idx, name in names.items():
        if str(name).lower() == lower_name:
            return int(idx)
    return None


def build_arg_parser() -> argparse.ArgumentParser:
    """Build CLI parser."""
    parser = argparse.ArgumentParser(description="Visualize YOLO examples for a class.")
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--class-name", default=None)
    parser.add_argument("--class-id", type=int, default=None)
    parser.add_argument("--split", required=True, choices=["train", "val"])
    parser.add_argument("--max-images", type=int, default=16)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    visualize_yolo_class_samples(args)


if __name__ == "__main__":
    main()

