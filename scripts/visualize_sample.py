"""Save a quick RGB/depth/GT visualization for one sample."""

import argparse
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from deep_oc_sort_3d.data.sample_dataset import SmartSpacesFrameDataset


def _draw_gt_boxes(rgb: np.ndarray, sample: Any) -> np.ndarray:
    image = rgb.copy()
    gt_objects = sample.get("gt_objects")
    camera_id = sample.get("camera_id")
    if gt_objects is None:
        return image

    for obj in gt_objects:
        bbox = obj.visible_bboxes_2d.get(camera_id)
        if bbox is None:
            continue
        x1, y1, x2, y2 = bbox
        cv2.rectangle(
            image,
            (int(round(x1)), int(round(y1))),
            (int(round(x2)), int(round(y2))),
            (255, 0, 0),
            2,
        )
    return image


def _depth_to_rgb(depth: np.ndarray, output_size: Any) -> np.ndarray:
    depth_float = depth.astype(np.float32)
    finite = np.isfinite(depth_float)
    if not finite.any():
        normalized = np.zeros(depth_float.shape, dtype=np.uint8)
    else:
        min_value = float(np.nanmin(depth_float[finite]))
        max_value = float(np.nanmax(depth_float[finite]))
        if max_value <= min_value:
            normalized = np.zeros(depth_float.shape, dtype=np.uint8)
        else:
            normalized = ((depth_float - min_value) / (max_value - min_value) * 255.0).clip(0, 255).astype(np.uint8)
    color_bgr = cv2.applyColorMap(normalized, cv2.COLORMAP_TURBO)
    color_rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)
    if output_size is not None:
        color_rgb = cv2.resize(color_rgb, output_size, interpolation=cv2.INTER_NEAREST)
    return color_rgb


def visualize_sample(args: Any) -> None:
    """Load one sample and save a PNG visualization."""
    dataset = SmartSpacesFrameDataset(
        root=args.root,
        split=args.split,
        scene_name=args.scene,
        max_frames=args.frame_index + 1,
        camera_id=args.camera_id,
        load_rgb=True,
        load_depth=True,
        load_gt=True,
        depth_dataset_name=args.depth_dataset_name,
    )

    sample = dataset[args.frame_index]
    rgb = sample.get("rgb")
    if rgb is None:
        print("RGB frame is None; no visualization saved.")
        for warning in sample.get("warnings", []):
            print("warning: %s" % warning)
        return

    rgb_with_boxes = _draw_gt_boxes(rgb, sample)
    depth = sample.get("depth")
    if depth is not None:
        height, width = rgb_with_boxes.shape[:2]
        depth_rgb = _depth_to_rgb(depth, (width, height))
        output_rgb = np.concatenate([rgb_with_boxes, depth_rgb], axis=1)
    else:
        output_rgb = rgb_with_boxes

    args.output.parent.mkdir(parents=True, exist_ok=True)
    output_bgr = cv2.cvtColor(output_rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(args.output), output_bgr)
    print("Saved visualization to %s" % args.output)


def build_arg_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser."""
    parser = argparse.ArgumentParser(description="Visualize one SmartSpaces sample.")
    parser.add_argument("--root", required=True, type=Path, help="Path to MTMC_Tracking_2026.")
    parser.add_argument("--split", required=True, choices=["train", "val", "test"], help="Dataset split.")
    parser.add_argument("--scene", required=True, help="Scene name, for example Warehouse_000.")
    parser.add_argument("--camera-id", required=True, help="Camera id, for example Camera_0000.")
    parser.add_argument("--frame-index", type=int, required=True, help="0-based frame index.")
    parser.add_argument("--output", required=True, type=Path, help="Output PNG path.")
    parser.add_argument("--depth-dataset-name", default=None, help="Optional internal HDF5 dataset name.")
    return parser


def main() -> None:
    """CLI entrypoint."""
    parser = build_arg_parser()
    args = parser.parse_args()
    visualize_sample(args)


if __name__ == "__main__":
    main()

