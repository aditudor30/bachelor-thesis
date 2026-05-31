"""Inspect SmartSpaces dataset structure without running models."""

import argparse
from pathlib import Path
from typing import Any, Dict

from deep_oc_sort_3d.data.calibration import parse_calibration_json_dict
from deep_oc_sort_3d.data.dataset_structure import (
    expected_scene_names,
    get_scene_paths,
    list_scenes,
    list_splits,
    validate_scene_structure,
)
from deep_oc_sort_3d.data.ground_truth import parse_ground_truth_json_dict
from deep_oc_sort_3d.utils.io_utils import load_json_if_exists


def _exists_text(value: bool) -> str:
    return "yes" if value else "no"


def _print_scene_report(root: Path, split: str, scene_name: str, sample_gt: bool) -> None:
    scene_paths = get_scene_paths(root, split, scene_name)
    report = validate_scene_structure(scene_paths, split)
    exists = report["exists"]

    print("  - %s: valid=%s" % (scene_name, report["is_valid"]))
    print("    videos_dir: %s" % _exists_text(exists["videos_dir"]))
    print("    depth_maps_dir: %s" % _exists_text(exists["depth_maps_dir"]))
    print("    ground_truth: %s" % _exists_text(exists["ground_truth_path"]))
    print("    calibration: %s" % _exists_text(exists["calibration_path"]))
    print("    map.png: %s" % _exists_text(exists["map_path"]))

    if report["missing_required"]:
        print("    missing required: %s" % ", ".join(report["missing_required"]))
    if report["missing_optional"]:
        print("    missing optional: %s" % ", ".join(report["missing_optional"]))

    calibration_data = load_json_if_exists(scene_paths.calibration_path)
    if isinstance(calibration_data, dict):
        print("    calibration top-level keys: %s" % ", ".join(sorted(calibration_data.keys())))
        calibrations = parse_calibration_json_dict(calibration_data)
        print("    parsed cameras: %d" % len(calibrations))

    if sample_gt and split in ("train", "val"):
        gt_data = load_json_if_exists(scene_paths.ground_truth_path)
        if isinstance(gt_data, dict):
            objects = parse_ground_truth_json_dict(_first_frames(gt_data, 1))
            if objects:
                obj = objects[0]
                print(
                    "    sample gt: frame=%d object_id=%d object_type=%s visible_cameras=%s"
                    % (obj.frame_id, obj.object_id, obj.object_type, sorted(obj.visible_bboxes_2d.keys()))
                )
            else:
                print("    sample gt: no parseable objects in first frame")


def _first_frames(data: Dict[str, Any], count: int) -> Dict[str, Any]:
    keys = sorted(data.keys(), key=lambda item: int(item) if str(item).isdigit() else 0)
    selected = keys[:count]
    return {key: data[key] for key in selected}


def inspect_dataset(root: Path, max_scenes_per_split: int, sample_gt: bool) -> None:
    """Print a lightweight dataset structure report."""
    found_splits = list_splits(root)
    print("Dataset root: %s" % root)
    print("Found splits: %s" % (", ".join(found_splits) if found_splits else "none"))

    for split in ("train", "val", "test"):
        scenes = list_scenes(root, split)
        expected = expected_scene_names(split)
        print("")
        print("[%s]" % split)
        print("  expected scenes: %s" % ", ".join(expected))
        print("  found scenes: %s" % (", ".join(scenes) if scenes else "none"))

        missing_expected = [scene for scene in expected if scene not in scenes]
        if missing_expected:
            print("  missing expected scenes: %s" % ", ".join(missing_expected))

        for scene_name in scenes[:max_scenes_per_split]:
            _print_scene_report(root, split, scene_name, sample_gt)


def build_arg_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser."""
    parser = argparse.ArgumentParser(description="Inspect SmartSpaces MTMC dataset structure.")
    parser.add_argument("--root", required=True, type=Path, help="Path to MTMC_Tracking_2026.")
    parser.add_argument("--max-scenes-per-split", type=int, default=3, help="Limit printed scenes per split.")
    parser.add_argument("--no-sample-gt", action="store_true", help="Do not print a ground-truth sample.")
    return parser


def main() -> None:
    """CLI entrypoint."""
    parser = build_arg_parser()
    args = parser.parse_args()
    inspect_dataset(args.root, args.max_scenes_per_split, not args.no_sample_gt)


if __name__ == "__main__":
    main()
