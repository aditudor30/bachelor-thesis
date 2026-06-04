"""Audit Step 15G full-camera pseudo-3D coverage."""

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from deep_oc_sort_3d.pseudo3d_fullcam.fullcam_coverage_audit import audit_fullcam_coverage, write_coverage_reports
from deep_oc_sort_3d.pseudo3d_fullcam.fullcam_discovery import (
    discover_required_camera_files,
    filter_fullcam_items,
    read_fullcam_items_json,
)


def run(args: Any) -> Dict[str, Any]:
    """Run coverage audit for full-camera pseudo-3D outputs."""
    cfg = _load_yaml(args.config)
    output_root = _output_root(cfg)
    items = read_fullcam_items_json(output_root / "discovery" / "required_camera_files.json")
    if not items:
        items = discover_required_camera_files(cfg)
    items = filter_fullcam_items(items, _csv_arg(args.subsets), _csv_arg(args.scenes), _csv_arg(args.camera_ids), args.max_cameras)
    summary = audit_fullcam_coverage(items, output_root, cfg, show_progress=args.progress)
    write_coverage_reports(summary, output_root)
    print("stabilized file coverage: %s" % summary.get("stabilized_file_coverage"))
    print("stabilized success rate: %s" % summary.get("success_rate_stabilized"))
    return summary


def _load_yaml(path: Path) -> Dict[str, Any]:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        return {}
    if data.get("include_config"):
        return _load_yaml(Path(data["include_config"]))
    return data


def _output_root(cfg: Dict[str, Any]) -> Path:
    return Path(cfg.get("step15g", {}).get("output_root", "output/pseudo3d/baseline_v2_pseudo3d_fullcam"))


def _csv_arg(value: Optional[str]) -> Optional[List[str]]:
    if not value:
        return None
    return [part.strip() for part in value.split(",") if part.strip()]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit Step 15G full-camera pseudo-3D coverage.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max-cameras", type=int, default=None)
    parser.add_argument("--subsets", default=None)
    parser.add_argument("--scenes", default=None)
    parser.add_argument("--camera-ids", default=None)
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

