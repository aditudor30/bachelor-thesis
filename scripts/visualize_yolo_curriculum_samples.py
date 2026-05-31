"""Visualize samples from a YOLO curriculum export."""

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import yaml

from deep_oc_sort_3d.detection2d.yolo_curriculum_manifest import read_curriculum_manifest
from deep_oc_sort_3d.detection2d.yolo_label_utils import read_yolo_label_file
from deep_oc_sort_3d.detection2d.yolo_visualization import draw_yolo_labels_on_image, make_image_grid


def visualize_yolo_curriculum_samples(args: Any) -> None:
    """Save a grid of curriculum images that match optional filters."""
    names = _load_names(args.dataset / "data.yaml")
    class_id = _resolve_class_id(args.class_id, args.class_name, names)
    rows = _load_candidate_rows(args.dataset)
    panels = []
    for row in rows:
        if not _row_matches(row, class_id, names, args.difficulty):
            continue
        image_path = _resolve_path(row["image_path"], args.dataset)
        label_path = _resolve_path(row["label_path"], args.dataset)
        image_bgr = cv2.imread(str(image_path))
        if image_bgr is None:
            continue
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        labels = read_yolo_label_file(label_path)
        panel = draw_yolo_labels_on_image(image_rgb, labels, names)
        _draw_caption(panel, row)
        panels.append(panel)
        if len(panels) >= args.max_images:
            break
    if not panels:
        print("No curriculum samples matched the requested filters.")
        return
    grid = make_image_grid(panels, cols=args.cols, cell_size=(args.cell_width, args.cell_height))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.output), cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
    print("Saved %s with %d panels" % (args.output, len(panels)))


def _load_candidate_rows(dataset: Path) -> List[Dict[str, Any]]:
    manifest = dataset / "curriculum_manifest.csv"
    if manifest.exists():
        return read_curriculum_manifest(manifest)
    rows = []
    for image_path in sorted((dataset / "images" / "train").glob("*.*")):
        rows.append(
            {
                "image_path": str(image_path),
                "label_path": str(dataset / "labels" / "train" / (image_path.stem + ".txt")),
                "class_counts": {},
                "difficulties": {},
                "scene_name": "",
                "camera_id": "",
                "frame_id": "",
            }
        )
    return rows


def _row_matches(
    row: Dict[str, Any],
    class_id: Optional[int],
    names: Dict[int, str],
    difficulty: Optional[str],
) -> bool:
    if difficulty is not None and int(row.get("difficulties", {}).get(difficulty, 0)) <= 0:
        return False
    if class_id is None:
        return True
    class_name = names.get(int(class_id), str(class_id))
    class_counts = row.get("class_counts", {})
    if class_counts:
        return int(class_counts.get(class_name, 0)) > 0
    labels = read_yolo_label_file(Path(str(row["label_path"])))
    return any(label.class_id == int(class_id) for label in labels)


def _resolve_path(value: Any, dataset: Path) -> Path:
    path = Path(str(value))
    if path.exists():
        return path
    if path.is_absolute():
        return path
    candidate = dataset / path
    if candidate.exists():
        return candidate
    return path


def _draw_caption(image_rgb: Any, row: Dict[str, Any]) -> None:
    text = "%s %s frame=%s" % (row.get("scene_name", ""), row.get("camera_id", ""), row.get("frame_id", ""))
    cv2.putText(image_rgb, text, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)


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
    parser = argparse.ArgumentParser(description="Visualize images from a YOLO curriculum export.")
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--class-name", default=None)
    parser.add_argument("--class-id", type=int, default=None)
    parser.add_argument("--difficulty", default=None)
    parser.add_argument("--max-images", type=int, default=16)
    parser.add_argument("--cols", type=int, default=4)
    parser.add_argument("--cell-width", type=int, default=320)
    parser.add_argument("--cell-height", type=int, default=240)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    visualize_yolo_curriculum_samples(args)


if __name__ == "__main__":
    main()
