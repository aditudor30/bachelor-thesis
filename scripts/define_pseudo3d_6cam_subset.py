"""Define and discover the pseudo-3D 6-camera experiment subset."""

import argparse
from pathlib import Path
from typing import Any, Dict

import yaml

from deep_oc_sort_3d.audit3d.audit3d_io import write_json
from deep_oc_sort_3d.experiments.sixcam_subset import (
    discover_sixcam_subset,
    write_frame_coverage_csv,
    write_sixcam_subset_csv,
    write_sixcam_subset_json,
)


def run(args: Any) -> Dict[str, Any]:
    """Discover and write the fixed 6cam subset."""
    cfg = _load_yaml(args.config)
    output_root = Path(cfg.get("experiment", {}).get("output_root", "output/baseline_v2_pseudo3d_6cam_comparison"))
    predictions_root = Path(cfg.get("paths", {}).get("pseudo3d_predictions_root", "output/pseudo3d/baseline_v2_pseudo3d_stabilized/predictions_stabilized"))
    subset_root = output_root / "subset_definition"
    items = discover_sixcam_subset(predictions_root)
    write_sixcam_subset_json(items, subset_root / "six_camera_subset.json")
    write_sixcam_subset_csv(items, subset_root / "six_camera_subset.csv")
    write_frame_coverage_csv(items, subset_root / "six_camera_frame_coverage.csv")
    summary = {
        "camera_count": len(items),
        "total_predictions": sum(int(item.num_predictions or 0) for item in items),
        "total_complete": sum(int(item.num_complete or 0) for item in items),
        "min_completion_rate": min([item.completion_rate for item in items if item.completion_rate is not None] or [None]),
    }
    write_json(summary, subset_root / "six_camera_subset_summary.json")
    print("sixcam cameras: %s" % summary.get("camera_count"))
    print("total predictions: %s" % summary.get("total_predictions"))
    return summary


def _load_yaml(path: Path) -> Dict[str, Any]:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        return {}
    if data.get("include_config"):
        return _load_yaml(Path(data["include_config"]))
    return data


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Define pseudo-3D 6cam subset.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--overwrite", action="store_true")
    progress_group = parser.add_mutually_exclusive_group()
    progress_group.add_argument("--progress", dest="progress", action="store_true")
    progress_group.add_argument("--no-progress", dest="progress", action="store_false")
    parser.set_defaults(progress=True)
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
