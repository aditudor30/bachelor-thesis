"""Evaluate robust depth sampling methods against GT 3D centers."""

import argparse
import csv
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from deep_oc_sort_3d.data.sample_dataset import SmartSpacesFrameDataset
from deep_oc_sort_3d.geometry.camera_geometry import camera_to_world, ensure_matrix, pixel_depth_to_camera_point
from deep_oc_sort_3d.geometry.depth_quality import compare_backprojection_to_gt, guess_depth_unit, summarize_depth_array
from deep_oc_sort_3d.geometry.depth_sampling import sample_depth_robust
from deep_oc_sort_3d.geometry.projection_3d import bbox_bottom_center
from deep_oc_sort_3d.training.target_builder import TrainingTargetBuilder


METHODS = [
    "center",
    "bottom_center",
    "bbox_median",
    "bbox_percentile_30",
    "lower_median",
    "lower_percentile_30",
    "center_median",
    "trimmed_median",
    "histogram_mode",
]


FIELDNAMES = [
    "scene_name",
    "frame_id",
    "camera_id",
    "object_id",
    "class_name",
    "class_id",
    "bbox_xyxy",
    "method",
    "depth_value_raw",
    "depth_value_converted",
    "guessed_unit",
    "gt_center_x",
    "gt_center_y",
    "gt_center_z",
    "backprojected_x",
    "backprojected_y",
    "backprojected_z",
    "backprojection_error",
]


def _convert_scalar_depth(depth_value: Optional[float], guessed_unit: str) -> Optional[float]:
    if depth_value is None:
        return None
    if guessed_unit == "millimeters_likely":
        return float(depth_value) / 1000.0
    return float(depth_value)


def _backproject(
    bbox_xyxy: Tuple[float, float, float, float],
    depth_value: Optional[float],
    calibration: Any,
) -> Optional[np.ndarray]:
    if depth_value is None or calibration is None:
        return None
    intrinsic = ensure_matrix(calibration.intrinsic_matrix, (3, 3))
    extrinsic = ensure_matrix(calibration.extrinsic_matrix)
    if intrinsic is None or extrinsic is None:
        return None
    u, v = bbox_bottom_center(bbox_xyxy)
    try:
        point_camera = pixel_depth_to_camera_point(u, v, depth_value, intrinsic)
        return camera_to_world(point_camera, extrinsic)
    except Exception:
        return None


def _row(
    sample: Dict[str, Any],
    target: Any,
    method: str,
    depth_raw: Optional[float],
    depth_converted: Optional[float],
    guessed_unit: str,
    backprojected: Optional[np.ndarray],
    error: Optional[float],
) -> Dict[str, Any]:
    if backprojected is None:
        bx = None
        by = None
        bz = None
    else:
        bx = float(backprojected[0])
        by = float(backprojected[1])
        bz = float(backprojected[2])
    return {
        "scene_name": sample.get("scene_name"),
        "frame_id": sample.get("frame_id"),
        "camera_id": sample.get("camera_id"),
        "object_id": target.object_id,
        "class_name": target.class_name,
        "class_id": target.class_id,
        "bbox_xyxy": "" if target.bbox_xyxy is None else list(target.bbox_xyxy),
        "method": method,
        "depth_value_raw": depth_raw,
        "depth_value_converted": depth_converted,
        "guessed_unit": guessed_unit,
        "gt_center_x": float(target.center_3d[0]),
        "gt_center_y": float(target.center_3d[1]),
        "gt_center_z": float(target.center_3d[2]),
        "backprojected_x": bx,
        "backprojected_y": by,
        "backprojected_z": bz,
        "backprojection_error": error,
    }


def _print_summary(rows: List[Dict[str, Any]]) -> None:
    print("")
    print("Summary by method:")
    for method in METHODS:
        method_rows = [row for row in rows if row["method"] == method]
        errors = [
            float(row["backprojection_error"])
            for row in method_rows
            if row["backprojection_error"] is not None
        ]
        count = len(method_rows)
        valid_count = len(errors)
        failure_rate = 0.0 if count == 0 else float(count - valid_count) / float(count)
        print("  %s:" % method)
        print("    count: %d" % count)
        print("    valid_count: %d" % valid_count)
        if errors:
            arr = np.asarray(errors, dtype=float)
            print("    mean_error: %.6f" % float(np.mean(arr)))
            print("    median_error: %.6f" % float(np.median(arr)))
            print("    p75_error: %.6f" % float(np.percentile(arr, 75)))
            print("    p90_error: %.6f" % float(np.percentile(arr, 90)))
        else:
            print("    mean_error: None")
            print("    median_error: None")
            print("    p75_error: None")
            print("    p90_error: None")
        print("    failure_rate: %.6f" % failure_rate)


def evaluate_depth_sampling(args: Any) -> None:
    """Evaluate sampling methods on a bounded frame window."""
    dataset = SmartSpacesFrameDataset(
        root=args.root,
        split=args.split,
        scene_name=args.scene,
        max_frames=args.max_frames,
        camera_id=args.camera_id,
        load_rgb=False,
        load_depth=True,
        load_gt=True,
        depth_dataset_name=args.depth_dataset_name,
    )
    builder = TrainingTargetBuilder(use_depth_backprojection=False)
    rows = []

    for idx in range(min(args.max_frames, len(dataset))):
        sample = dataset[idx]
        depth = sample.get("depth")
        if depth is None:
            continue
        guessed_unit = guess_depth_unit(summarize_depth_array(depth, sample_name="frame_%d" % idx))
        frame_targets = builder.build_targets_from_sample(sample)
        for target in frame_targets.targets:
            if target.bbox_xyxy is None:
                continue
            for method in METHODS:
                depth_raw = sample_depth_robust(depth, target.bbox_xyxy, method=method)
                depth_converted = _convert_scalar_depth(depth_raw, guessed_unit)
                backprojected = _backproject(target.bbox_xyxy, depth_converted, sample.get("calibration"))
                error = compare_backprojection_to_gt(backprojected, target.center_3d)
                rows.append(
                    _row(
                        sample=sample,
                        target=target,
                        method=method,
                        depth_raw=depth_raw,
                        depth_converted=depth_converted,
                        guessed_unit=guessed_unit,
                        backprojected=backprojected,
                        error=error,
                    )
                )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print("Wrote %d rows to %s" % (len(rows), args.output))
    _print_summary(rows)


def build_arg_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser."""
    parser = argparse.ArgumentParser(description="Evaluate depth sampling methods.")
    parser.add_argument("--root", required=True, type=Path, help="Path to MTMC_Tracking_2026.")
    parser.add_argument("--split", required=True, choices=["train", "val"], help="Dataset split with depth and GT.")
    parser.add_argument("--scene", required=True, help="Scene name.")
    parser.add_argument("--camera-id", required=True, help="Camera id.")
    parser.add_argument("--max-frames", type=int, default=20, help="Number of frames to inspect.")
    parser.add_argument("--output", required=True, type=Path, help="Output CSV path.")
    parser.add_argument("--depth-dataset-name", default=None, help="Optional internal HDF5 dataset name.")
    return parser


def main() -> None:
    """CLI entrypoint."""
    parser = build_arg_parser()
    args = parser.parse_args()
    evaluate_depth_sampling(args)


if __name__ == "__main__":
    main()

