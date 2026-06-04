"""Discover Step 15G full-camera pseudo-3D inputs."""

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from deep_oc_sort_3d.audit3d.audit3d_io import progress_iter, write_csv, write_json
from deep_oc_sort_3d.pseudo3d_fullcam.fullcam_discovery import (
    audit_existing_predictions,
    discover_required_camera_files,
    filter_fullcam_items,
    missing_prediction_rows,
    write_fullcam_items_csv,
    write_fullcam_items_json,
)


def run(args: Any) -> Dict[str, Any]:
    """Run full-camera input discovery."""
    cfg = _load_yaml(args.config)
    output_root = _output_root(cfg)
    items = discover_required_camera_files(cfg)
    items = filter_fullcam_items(items, _csv_arg(args.subsets), _csv_arg(args.scenes), _csv_arg(args.camera_ids), args.max_cameras)
    items = list(progress_iter(items, args.progress, "discover fullcam inputs", "camera"))
    discovery_root = output_root / "discovery"
    write_fullcam_items_json(items, discovery_root / "required_camera_files.json")
    write_fullcam_items_csv(items, discovery_root / "required_camera_files.csv")
    audit = audit_existing_predictions(items)
    write_json(audit, discovery_root / "existing_raw_predictions_audit.json")
    write_json(audit, discovery_root / "existing_stabilized_predictions_audit.json")
    write_csv(missing_prediction_rows(items), discovery_root / "missing_camera_files.csv")
    print("required camera files: %s" % len(items))
    print("raw existing: %s" % audit.get("raw_prediction_files_existing"))
    print("stabilized existing: %s" % audit.get("stabilized_prediction_files_existing"))
    return audit


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
    parser = argparse.ArgumentParser(description="Discover Step 15G full-camera pseudo-3D inputs.")
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

