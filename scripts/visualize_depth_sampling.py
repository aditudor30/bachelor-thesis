"""Visualize robust depth sampling regions for one frame."""

import argparse
from pathlib import Path
from typing import Any, List, Tuple

import cv2
import numpy as np

from deep_oc_sort_3d.data.sample_dataset import SmartSpacesFrameDataset
from deep_oc_sort_3d.geometry.depth_sampling import sample_depth_robust
from deep_oc_sort_3d.geometry.projection_3d import bbox_bottom_center, bbox_center
from deep_oc_sort_3d.training.target_builder import TrainingTargetBuilder


def _depth_to_rgb(depth: np.ndarray, output_size: Tuple[int, int]) -> np.ndarray:
    depth_float = depth.astype(np.float32)
    finite = np.isfinite(depth_float) & (depth_float > 0)
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
    return cv2.resize(color_rgb, output_size, interpolation=cv2.INTER_NEAREST)


def _lower_region_box(bbox_xyxy: Tuple[float, float, float, float], lower_ratio: float) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox_xyxy
    y_lower = float(y1) + (float(y2) - float(y1)) * (1.0 - float(lower_ratio))
    return (int(round(x1)), int(round(y_lower)), int(round(x2)), int(round(y2)))


def _draw_targets(image: np.ndarray, targets: List[Any], annotate_depth: bool) -> np.ndarray:
    out = image.copy()
    for index, target in enumerate(targets):
        if target.bbox_xyxy is None:
            continue
        x1, y1, x2, y2 = target.bbox_xyxy
        p1 = (int(round(x1)), int(round(y1)))
        p2 = (int(round(x2)), int(round(y2)))
        cv2.rectangle(out, p1, p2, (255, 0, 0), 2)

        cx, cy = bbox_center(target.bbox_xyxy)
        bx, by = bbox_bottom_center(target.bbox_xyxy)
        cv2.circle(out, (int(round(cx)), int(round(cy))), 4, (0, 255, 255), -1)
        cv2.circle(out, (int(round(bx)), int(round(by))), 4, (255, 0, 255), -1)

        lower = _lower_region_box(target.bbox_xyxy, 0.35)
        cv2.rectangle(out, (lower[0], lower[1]), (lower[2], lower[3]), (0, 255, 0), 2)

        label = "%s:%d" % (target.class_name, target.object_id)
        if annotate_depth and index < 8:
            label = "%s lm=%s bc=%s" % (
                label,
                _short_depth(target.extra_lower_median),
                _short_depth(target.extra_bottom_center),
            )
        cv2.putText(out, label, p1, cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 0, 0), 2)
    return out


def _short_depth(value: Any) -> str:
    if value is None:
        return "None"
    return "%.1f" % float(value)


def visualize_depth_sampling(args: Any) -> None:
    """Save a composed image showing sampling points and regions."""
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
    depth = sample.get("depth")
    if rgb is None:
        print("RGB is None; cannot save visualization.")
        return
    if depth is None:
        print("Depth is None; saving RGB with bbox only.")

    builder = TrainingTargetBuilder(use_depth_backprojection=False)
    frame_targets = builder.build_targets_from_sample(sample)
    visible_targets = [target for target in frame_targets.targets if target.bbox_xyxy is not None]
    for target in visible_targets:
        if depth is None:
            target.extra_lower_median = None
            target.extra_bottom_center = None
        else:
            target.extra_lower_median = sample_depth_robust(depth, target.bbox_xyxy, method="lower_median")
            target.extra_bottom_center = sample_depth_robust(depth, target.bbox_xyxy, method="bottom_center")

    rgb_bbox = _draw_targets(rgb, visible_targets, annotate_depth=False)
    sampling_panel = _draw_targets(rgb, visible_targets, annotate_depth=True)
    panels = [rgb_bbox]
    if depth is not None:
        height, width = rgb.shape[:2]
        depth_rgb = _depth_to_rgb(depth, (width, height))
        depth_panel = _draw_targets(depth_rgb, visible_targets, annotate_depth=False)
        panels.append(depth_panel)
    panels.append(sampling_panel)

    output_rgb = np.concatenate(panels, axis=1)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.output), cv2.cvtColor(output_rgb, cv2.COLOR_RGB2BGR))
    print("Saved depth sampling visualization to %s" % args.output)


def build_arg_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser."""
    parser = argparse.ArgumentParser(description="Visualize robust depth sampling diagnostics.")
    parser.add_argument("--root", required=True, type=Path, help="Path to MTMC_Tracking_2026.")
    parser.add_argument("--split", required=True, choices=["train", "val"], help="Dataset split.")
    parser.add_argument("--scene", required=True, help="Scene name.")
    parser.add_argument("--camera-id", required=True, help="Camera id.")
    parser.add_argument("--frame-index", type=int, required=True, help="0-based frame index.")
    parser.add_argument("--output", required=True, type=Path, help="Output PNG path.")
    parser.add_argument("--depth-dataset-name", default=None, help="Optional internal HDF5 dataset name.")
    return parser


def main() -> None:
    """CLI entrypoint."""
    parser = build_arg_parser()
    args = parser.parse_args()
    visualize_depth_sampling(args)


if __name__ == "__main__":
    main()

