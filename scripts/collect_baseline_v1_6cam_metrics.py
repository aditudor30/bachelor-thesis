"""Collect read-only baseline_v1 metrics for the 6-camera subset."""

import argparse
from pathlib import Path
from typing import Any, Dict

import yaml

from deep_oc_sort_3d.audit3d.audit3d_io import write_csv, write_json
from deep_oc_sort_3d.experiments.baseline_subset_metrics import collect_all_subset_metrics
from deep_oc_sort_3d.experiments.sixcam_subset import read_sixcam_subset_json, sixcam_items_from_config


def run(args: Any) -> Dict[str, Any]:
    cfg = _load_yaml(args.config)
    output_root = Path(cfg.get("experiment", {}).get("output_root", "output/baseline_v2_pseudo3d_6cam_comparison"))
    items = _items(cfg, output_root)
    paths = _v1_paths(cfg)
    metrics = collect_all_subset_metrics(paths, items, "baseline_v1_geometry_only_6cam")
    _write_metric_bundle(metrics, output_root / "v1_subset_metrics")
    print("v1 observations: %s" % metrics.get("observations", {}).get("num_observations"))
    return metrics


def _items(cfg: Dict[str, Any], output_root: Path) -> Any:
    subset_path = output_root / "subset_definition" / "six_camera_subset.json"
    items = read_sixcam_subset_json(subset_path)
    return items if items else sixcam_items_from_config(cfg)


def _v1_paths(cfg: Dict[str, Any]) -> Dict[str, Any]:
    paths = cfg.get("paths", {})
    return {
        "pipeline_root": paths.get("v1_pipeline_root"),
        "local_tracks_root": paths.get("v1_local_tracks_root"),
        "tracklets_root": paths.get("v1_tracklets_root"),
        "candidates_root": paths.get("v1_candidates_root"),
        "motion_clean_root": paths.get("v1_motion_clean_root"),
        "global_root": paths.get("v1_global_root"),
        "final_export_root": paths.get("v1_final_export_root"),
    }


def _write_metric_bundle(metrics: Dict[str, Any], root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    mapping = {
        "observations": "observations_summary",
        "local_tracking": "local_tracking_summary",
        "tracklets": "tracklets_summary",
        "candidates": "candidates_summary",
        "motion_clean": "motion_clean_summary",
        "global_association": "global_association_summary",
        "final_export": "final_export_summary",
    }
    for key, name in mapping.items():
        section = metrics.get(key, {})
        write_json(section, root / ("%s.json" % name))
        write_csv(_rows(section), root / ("%s.csv" % name))
    write_json(metrics, root / "track1_like_summary.json")


def _rows(section: Dict[str, Any]) -> Any:
    rows = section.get("rows", [])
    return rows if isinstance(rows, list) else []


def _load_yaml(path: Path) -> Dict[str, Any]:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        return {}
    if data.get("include_config"):
        return _load_yaml(Path(data["include_config"]))
    return data


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Collect baseline_v1 6cam metrics.")
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
