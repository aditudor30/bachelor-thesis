"""Visualize RGB-depth-GT-calibration alignment for one frame."""

import argparse
from pathlib import Path
from typing import Any, List, Tuple

import cv2
import numpy as np

from deep_oc_sort_3d.data.sample_dataset import SmartSpacesFrameDataset
from deep_oc_sort_3d.geometry.box_projection import project_3d_box_to_image
from deep_oc_sort_3d.geometry.camera_geometry import project_world_to_image
from deep_oc_sort_3d.training.target_builder import TrainingTargetBuilder


BOX_EDGES = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 0),
    (4, 5),
    (5, 6),
    (6, 7),
    (7, 4),
    (0, 4),
    (1, 5),
    (2, 6),
    (3, 7),
]


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


def _draw_bbox_and_labels(image: np.ndarray, targets: List[Any]) -> np.ndarray:
    out = image.copy()
    for target in targets:
        if target.bbox_xyxy is None:
            continue
        x1, y1, x2, y2 = target.bbox_xyxy
        p1 = (int(round(x1)), int(round(y1)))
        p2 = (int(round(x2)), int(round(y2)))
        cv2.rectangle(out, p1, p2, (255, 0, 0), 2)
        label = "%s:%d" % (target.class_name, target.object_id)
        cv2.putText(out, label, p1, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    return out


def _draw_projected_geometry(image: np.ndarray, targets: List[Any], calibration: Any) -> np.ndarray:
    out = image.copy()
    if calibration is None:
        return out

    for target in targets:
        box_2d = project_3d_box_to_image(target.center_3d, target.dimensions_3d, target.yaw, calibration)
        if box_2d is not None:
            for edge in BOX_EDGES:
                p1 = tuple(np.round(box_2d[edge[0]]).astype(int))
                p2 = tuple(np.round(box_2d[edge[1]]).astype(int))
                cv2.line(out, p1, p2, (0, 255, 0), 2)

        center_2d = project_world_to_image(
            target.center_3d,
            camera_matrix=calibration.camera_matrix,
            intrinsic_matrix=calibration.intrinsic_matrix,
            extrinsic_matrix=calibration.extrinsic_matrix,
        )
        if center_2d is not None:
            cv2.circle(out, tuple(np.round(center_2d).astype(int)), 5, (255, 255, 0), -1)

        if target.backprojected_center_3d is not None:
            backprojected_2d = project_world_to_image(
                target.backprojected_center_3d,
                camera_matrix=calibration.camera_matrix,
                intrinsic_matrix=calibration.intrinsic_matrix,
                extrinsic_matrix=calibration.extrinsic_matrix,
            )
            if backprojected_2d is not None:
                cv2.circle(out, tuple(np.round(backprojected_2d).astype(int)), 5, (255, 0, 255), -1)
    return out


def visualize_alignment(args: Any) -> None:
    """Save one alignment visualization image."""
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
        print("RGB is None; cannot save alignment visualization.")
        return

    builder = TrainingTargetBuilder()
    frame_targets = builder.build_targets_from_sample(sample)

    if args.split == "test":
        print("Test split has no depth and no ground truth; saving RGB-only debug image.")
        panels = [rgb]
    else:
        bbox_panel = _draw_bbox_and_labels(rgb, frame_targets.targets)
        projected_panel = _draw_projected_geometry(rgb, frame_targets.targets, sample.get("calibration"))
        panels = [bbox_panel]

        depth = sample.get("depth")
        if depth is not None:
            height, width = rgb.shape[:2]
            panels.append(_depth_to_rgb(depth, (width, height)))
        panels.append(projected_panel)

    output_rgb = np.concatenate(panels, axis=1)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output_bgr = cv2.cvtColor(output_rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(args.output), output_bgr)
    print("Saved alignment visualization to %s" % args.output)


def build_arg_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser."""
    parser = argparse.ArgumentParser(description="Visualize SmartSpaces alignment for one frame.")
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
    visualize_alignment(args)


if __name__ == "__main__":
    main()
