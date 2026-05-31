"""Visualize bbox audit examples by class and difficulty."""

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from deep_oc_sort_3d.data.sample_dataset import SmartSpacesFrameDataset
from deep_oc_sort_3d.detection2d.yolo_bbox_audit import load_bbox_audit_csv
from deep_oc_sort_3d.detection2d.yolo_visualization import make_image_grid


def visualize_bbox_quality_samples(args: Any) -> None:
    """Save a grid of RGB frames with selected audit bboxes drawn."""
    rows = load_bbox_audit_csv(args.audit_csv)
    selected = _select_rows(rows, args.class_name, args.class_id, args.difficulty)
    panels = []
    dataset_cache = {}
    for row in selected[: args.max_images]:
        key = (row["split"], row["scene_name"], row["camera_id"])
        if key not in dataset_cache:
            dataset_cache[key] = SmartSpacesFrameDataset(
                root=args.root,
                split=row["split"],
                scene_name=row["scene_name"],
                camera_id=row["camera_id"],
                load_rgb=True,
                load_depth=False,
                load_gt=False,
            )
        dataset = dataset_cache[key]
        frame_id = int(row["frame_id"])
        if frame_id < 0 or frame_id >= len(dataset):
            continue
        sample = dataset[frame_id]
        image = sample.get("rgb")
        if image is None:
            continue
        panel = image.copy()
        _draw_row(panel, row)
        panels.append(panel)
    if not panels:
        print("No matching samples found.")
        return
    grid = make_image_grid(panels, cols=4)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.output), cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
    print("Saved %s with %d panels" % (args.output, len(panels)))


def _select_rows(
    rows: List[Dict[str, Any]],
    class_name: Optional[str],
    class_id: Optional[int],
    difficulty: str,
) -> List[Dict[str, Any]]:
    selected = []
    for row in rows:
        if class_name is not None and row["class_name"] != class_name:
            continue
        if class_id is not None and int(row["class_id"]) != int(class_id):
            continue
        if difficulty != "all" and row["difficulty"] != difficulty:
            continue
        selected.append(row)
    return sorted(selected, key=lambda item: float(item["area_norm"]), reverse=True)


def _draw_row(image: np.ndarray, row: Dict[str, Any]) -> None:
    bbox = (float(row["x1"]), float(row["y1"]), float(row["x2"]), float(row["y2"]))
    label = "%s id=%s area=%.4f %s" % (
        row["class_name"],
        row["object_id"],
        float(row["area_norm"]),
        row["difficulty"],
    )
    x1, y1, x2, y2 = bbox
    p1 = (int(round(x1)), int(round(y1)))
    p2 = (int(round(x2)), int(round(y2)))
    cv2.rectangle(image, p1, p2, (0, 255, 0), 2)
    cv2.putText(image, label, p1, cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Visualize bbox quality audit examples.")
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--audit-csv", required=True, type=Path)
    parser.add_argument("--class-name", default=None)
    parser.add_argument("--class-id", type=int, default=None)
    parser.add_argument("--difficulty", default="all", choices=["easy", "medium", "hard", "all"])
    parser.add_argument("--max-images", type=int, default=16)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    visualize_bbox_quality_samples(args)


if __name__ == "__main__":
    main()

