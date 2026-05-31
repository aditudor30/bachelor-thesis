"""Audit class distribution in an exported YOLO dataset."""

import argparse
from pathlib import Path
from typing import Any

from deep_oc_sort_3d.detection2d.yolo_class_audit import (
    count_yolo_labels_by_class,
    summarize_class_distribution,
    top_images_with_classes,
)


def audit_yolo_labels_by_class(args: Any) -> None:
    """Print YOLO label class counts and rare-class examples."""
    counts = count_yolo_labels_by_class(args.dataset, args.split)
    print(summarize_class_distribution(counts))
    print("images: %d" % counts.get("num_images", 0))
    print("images_with_labels: %d" % counts.get("images_with_labels", 0))
    print("empty_images: %d" % counts.get("empty_images", 0))
    print("missing_label_files: %d" % counts.get("missing_label_files", 0))
    print("names: %s" % counts.get("names", {}))
    print("")
    examples = top_images_with_classes(counts, args.target_classes, limit=args.top_k)
    print("Top images with target classes (%s):" % ", ".join(args.target_classes))
    if not examples:
        print("  none")
    for image_path, class_counts in examples:
        print("  %s %s" % (image_path, class_counts))


def build_arg_parser() -> argparse.ArgumentParser:
    """Build CLI parser."""
    parser = argparse.ArgumentParser(description="Audit YOLO labels by class.")
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--split", required=True, choices=["train", "val"])
    parser.add_argument("--target-classes", nargs="+", default=["Forklift", "PalletTruck"])
    parser.add_argument("--top-k", type=int, default=10)
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    audit_yolo_labels_by_class(args)


if __name__ == "__main__":
    main()

