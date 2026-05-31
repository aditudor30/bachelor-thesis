"""Inspect an exported YOLO dataset."""

import argparse
from pathlib import Path
from typing import Any, Dict, List

import yaml

from deep_oc_sort_3d.detection2d.yolo_label_utils import read_yolo_label_file


def inspect_yolo_dataset(args: Any) -> None:
    """Print dataset integrity checks."""
    dataset = args.dataset
    required = [
        dataset / "images" / "train",
        dataset / "images" / "val",
        dataset / "labels" / "train",
        dataset / "labels" / "val",
        dataset / "data.yaml",
    ]
    for path in required:
        print("%s exists=%s" % (path, path.exists()))

    class_names = _load_names(dataset / "data.yaml")
    for split in ("train", "val"):
        images = sorted((dataset / "images" / split).glob("*.*"))
        labels_dir = dataset / "labels" / split
        missing_labels = []
        invalid_labels = []
        class_counts = {}
        examples = []
        for image_path in images:
            label_path = labels_dir / (image_path.stem + ".txt")
            if not label_path.exists():
                missing_labels.append(str(label_path))
                continue
            labels = read_yolo_label_file(label_path)
            if len(examples) < 5:
                examples.append((str(label_path), labels[:3]))
            for label in labels:
                if label.class_id not in class_names:
                    invalid_labels.append((str(label_path), "unknown_class", label.class_id))
                values = [label.x_center_norm, label.y_center_norm, label.width_norm, label.height_norm]
                if any(value < 0.0 or value > 1.0 for value in values):
                    invalid_labels.append((str(label_path), "out_of_range", values))
                if label.width_norm <= 0.0 or label.height_norm <= 0.0:
                    invalid_labels.append((str(label_path), "non_positive_size", values))
                class_counts[label.class_id] = class_counts.get(label.class_id, 0) + 1
        print("")
        print("[%s]" % split)
        print("images: %d" % len(images))
        print("missing label files: %d" % len(missing_labels))
        print("invalid labels: %d" % len(invalid_labels))
        print("class counts: %s" % class_counts)
        print("examples: %s" % examples)


def _load_names(path: Path) -> Dict[int, str]:
    if not path.exists():
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    names = data.get("names", {}) if isinstance(data, dict) else {}
    parsed = {}
    for key, value in names.items():
        parsed[int(key)] = str(value)
    return parsed


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inspect exported YOLO dataset.")
    parser.add_argument("--dataset", required=True, type=Path)
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    inspect_yolo_dataset(args)


if __name__ == "__main__":
    main()

