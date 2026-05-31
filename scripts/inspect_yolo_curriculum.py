"""Inspect a YOLO curriculum export."""

import argparse
from pathlib import Path
from typing import Any, Dict, List

import yaml

from deep_oc_sort_3d.detection2d.yolo_curriculum_manifest import check_manifest_for_duplicates, read_curriculum_manifest
from deep_oc_sort_3d.detection2d.yolo_label_utils import read_yolo_label_file


def inspect_yolo_curriculum(args: Any) -> None:
    """Print integrity and distribution checks for a curriculum export."""
    dataset = args.dataset
    data_yaml = dataset / "data.yaml"
    manifest_path = dataset / "curriculum_manifest.csv"
    summary_path = dataset / "curriculum_summary.json"
    required = [
        dataset / "images" / "train",
        dataset / "labels" / "train",
        data_yaml,
        manifest_path,
        summary_path,
    ]
    for path in required:
        print("%s exists=%s" % (path, path.exists()))

    names = _load_names(data_yaml)
    images = sorted((dataset / "images" / "train").glob("*.*"))
    labels_dir = dataset / "labels" / "train"
    missing_labels = []
    invalid_labels = []
    class_counts = {}
    for image_path in images:
        label_path = labels_dir / (image_path.stem + ".txt")
        if not label_path.exists():
            missing_labels.append(str(label_path))
            continue
        labels = read_yolo_label_file(label_path)
        for label in labels:
            values = [label.x_center_norm, label.y_center_norm, label.width_norm, label.height_norm]
            if label.class_id not in names:
                invalid_labels.append((str(label_path), "unknown_class", label.class_id))
            if any(value < 0.0 or value > 1.0 for value in values):
                invalid_labels.append((str(label_path), "out_of_range", values))
            if label.width_norm <= 0.0 or label.height_norm <= 0.0:
                invalid_labels.append((str(label_path), "non_positive_size", values))
            class_counts[label.class_id] = class_counts.get(label.class_id, 0) + 1

    print("")
    print("images: %d" % len(images))
    print("missing label files: %d" % len(missing_labels))
    print("invalid labels: %d" % len(invalid_labels))
    print("class counts by id: %s" % class_counts)
    print("class counts by name: %s" % _counts_by_name(class_counts, names))

    if manifest_path.exists():
        rows = read_curriculum_manifest(manifest_path)
        duplicate_report = check_manifest_for_duplicates(manifest_path)
        print("")
        print("manifest rows: %d" % len(rows))
        print("duplicate frames: %d" % duplicate_report["num_duplicates"])
        print("manifest per scene: %s" % _count_rows(rows, "scene_name"))
        print("manifest per camera: %s" % _count_rows(rows, "camera_id"))
        print("manifest per difficulty: %s" % _difficulty_counts(rows))
        print("person-only frames: %d" % len([row for row in rows if row["contains_person_only"]]))
        print("rare-class frames: %d" % len([row for row in rows if row["contains_rare_class"]]))
        missing_classes = [name for class_id, name in names.items() if class_counts.get(class_id, 0) == 0]
        if missing_classes:
            print("warning: missing classes in labels: %s" % ", ".join(missing_classes))


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


def _counts_by_name(class_counts: Dict[int, int], names: Dict[int, str]) -> Dict[str, int]:
    counts = {}
    for class_id, count in class_counts.items():
        counts[names.get(int(class_id), str(class_id))] = int(count)
    return counts


def _count_rows(rows: List[Dict[str, Any]], field: str) -> Dict[str, int]:
    counts = {}
    for row in rows:
        key = str(row.get(field))
        counts[key] = counts.get(key, 0) + 1
    return counts


def _difficulty_counts(rows: List[Dict[str, Any]]) -> Dict[str, int]:
    counts = {}
    for row in rows:
        for difficulty, count in row.get("difficulties", {}).items():
            counts[difficulty] = counts.get(difficulty, 0) + int(count)
    return counts


def build_arg_parser() -> argparse.ArgumentParser:
    """Build CLI parser."""
    parser = argparse.ArgumentParser(description="Inspect a YOLO curriculum dataset.")
    parser.add_argument("--dataset", required=True, type=Path)
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    inspect_yolo_curriculum(args)


if __name__ == "__main__":
    main()
