"""Export simplified training targets for a small frame window."""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from deep_oc_sort_3d.data.sample_dataset import SmartSpacesFrameDataset
from deep_oc_sort_3d.training.target_builder import TrainingTargetBuilder


def _array_to_list(value: Optional[np.ndarray]) -> Optional[List[float]]:
    if value is None:
        return None
    return [float(item) for item in np.asarray(value, dtype=float).reshape(-1)]


def _target_to_dict(target: Any) -> Dict[str, Any]:
    return {
        "frame_id": target.frame_id,
        "camera_id": target.camera_id,
        "object_id": target.object_id,
        "class_name": target.class_name,
        "class_id": target.class_id,
        "bbox_xyxy": None if target.bbox_xyxy is None else [float(item) for item in target.bbox_xyxy],
        "center_3d": _array_to_list(target.center_3d),
        "dimensions_3d": _array_to_list(target.dimensions_3d),
        "yaw": float(target.yaw),
        "depth_value": None if target.depth_value is None else float(target.depth_value),
        "backprojection_error": None
        if target.backprojection_error is None
        else float(target.backprojection_error),
    }


def debug_targets(args: Any) -> None:
    """Write simplified targets to a JSONL file."""
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
    builder = TrainingTargetBuilder()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with args.output.open("w", encoding="utf-8") as handle:
        for idx in range(min(args.max_frames, len(dataset))):
            sample = dataset[idx]
            frame_targets = builder.build_targets_from_sample(sample)
            for target in frame_targets.targets:
                handle.write(json.dumps(_target_to_dict(target), sort_keys=True) + "\n")
                written += 1
    print("Wrote %d target rows to %s" % (written, args.output))


def build_arg_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser."""
    parser = argparse.ArgumentParser(description="Export SmartSpaces training target debug JSONL.")
    parser.add_argument("--root", required=True, type=Path, help="Path to MTMC_Tracking_2026.")
    parser.add_argument("--split", required=True, choices=["train", "val", "test"], help="Dataset split.")
    parser.add_argument("--scene", required=True, help="Scene name, for example Warehouse_000.")
    parser.add_argument("--camera-id", required=True, help="Camera id, for example Camera_0000.")
    parser.add_argument("--max-frames", type=int, default=10, help="Small frame count to export.")
    parser.add_argument("--output", required=True, type=Path, help="Output JSONL path.")
    parser.add_argument("--depth-dataset-name", default=None, help="Optional internal HDF5 dataset name.")
    return parser


def main() -> None:
    """CLI entrypoint."""
    parser = build_arg_parser()
    args = parser.parse_args()
    debug_targets(args)


if __name__ == "__main__":
    main()

