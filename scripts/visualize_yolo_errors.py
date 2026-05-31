"""Visualize YOLO error cases from a confusion diagnostics JSON."""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2

from deep_oc_sort_3d.data.sample_dataset import SmartSpacesFrameDataset


def visualize_yolo_errors(args: Any) -> None:
    """Save debug PNGs for false positives, false negatives, and confusions."""
    data = json.loads(args.confusions.read_text(encoding="utf-8"))
    rows = []
    for key in ["false_positives", "false_negatives", "class_confusions"]:
        rows.extend(data.get(key, []))
    rows = [
        row
        for row in rows
        if row.get("scene_name") == args.scene
        and row.get("camera_id") == args.camera_id
        and row.get("split") == args.split
    ][: args.max_images]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    dataset = SmartSpacesFrameDataset(
        root=args.root,
        split=args.split,
        scene_name=args.scene,
        camera_id=args.camera_id,
        load_rgb=True,
        load_depth=False,
        load_gt=False,
    )
    saved = 0
    for idx, row in enumerate(rows):
        frame_id = int(row["frame_id"])
        if frame_id < 0 or frame_id >= len(dataset):
            continue
        sample = dataset[frame_id]
        image = sample.get("rgb")
        if image is None:
            continue
        drawn = image.copy()
        _draw_error(drawn, row)
        out_path = args.output_dir / ("%03d_%s_%06d.png" % (idx, row.get("type", "error"), frame_id))
        cv2.imwrite(str(out_path), cv2.cvtColor(drawn, cv2.COLOR_RGB2BGR))
        saved += 1
    print("Saved %d images to %s" % (saved, args.output_dir))


def _draw_error(image: Any, row: Dict[str, Any]) -> None:
    pred_box = _box(row.get("bbox_xyxy"))
    gt_box = _box(row.get("gt_bbox_xyxy"))
    if pred_box is not None:
        _draw_box(image, pred_box, "pred %s %.2f" % (row.get("pred_class_name"), _float(row.get("confidence"))), (255, 0, 0))
    if gt_box is not None:
        _draw_box(image, gt_box, "gt %s" % row.get("gt_class_name"), (0, 255, 0))
    text = "%s IoU=%s" % (row.get("type"), row.get("iou"))
    cv2.putText(image, text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)


def _draw_box(image: Any, bbox: Tuple[float, float, float, float], label: str, color: Tuple[int, int, int]) -> None:
    x1, y1, x2, y2 = bbox
    p1 = (int(round(x1)), int(round(y1)))
    p2 = (int(round(x2)), int(round(y2)))
    cv2.rectangle(image, p1, p2, color, 2)
    cv2.putText(image, label, p1, cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)


def _box(value: Any) -> Optional[Tuple[float, float, float, float]]:
    if value is None or len(value) < 4:
        return None
    return (float(value[0]), float(value[1]), float(value[2]), float(value[3]))


def _float(value: Any) -> float:
    if value is None:
        return 0.0
    return float(value)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Visualize YOLO error cases.")
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--split", required=True, choices=["train", "val"])
    parser.add_argument("--scene", required=True)
    parser.add_argument("--detections-dir", required=True, type=Path)
    parser.add_argument("--camera-id", required=True)
    parser.add_argument("--confusions", required=True, type=Path)
    parser.add_argument("--max-images", type=int, default=20)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    visualize_yolo_errors(args)


if __name__ == "__main__":
    main()

