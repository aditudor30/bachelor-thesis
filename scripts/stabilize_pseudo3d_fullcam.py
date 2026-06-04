"""Stabilize full-camera pseudo-3D predictions for Step 15G."""

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from deep_oc_sort_3d.pseudo3d_fullcam.fullcam_discovery import (
    discover_required_camera_files,
    filter_fullcam_items,
    read_fullcam_items_json,
)
from deep_oc_sort_3d.pseudo3d_fullcam.fullcam_stabilization import stabilize_items


def run(args: Any) -> Dict[str, Any]:
    """Run full-camera pseudo-3D stabilization."""
    cfg = _load_yaml(args.config)
    _apply_stabilization_flags(cfg, args)
    output_root = _output_root(cfg)
    items = read_fullcam_items_json(output_root / "discovery" / "required_camera_files.json")
    if not items:
        items = discover_required_camera_files(cfg)
    items = filter_fullcam_items(items, _csv_arg(args.subsets), _csv_arg(args.scenes), _csv_arg(args.camera_ids), args.max_cameras)
    summary = stabilize_items(items, cfg, show_progress=args.progress)
    print("stabilized camera files: %s" % summary.get("camera_count"))
    print("stabilized success rate: %s" % summary.get("success_rate"))
    return summary


def _apply_stabilization_flags(cfg: Dict[str, Any], args: Any) -> None:
    section = cfg.setdefault("stabilization", {})
    if args.overwrite:
        section["overwrite"] = True
        section["skip_existing"] = False
    if args.skip_existing is not None:
        section["skip_existing"] = bool(args.skip_existing)


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
    parser = argparse.ArgumentParser(description="Stabilize full-camera pseudo-3D predictions.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max-cameras", type=int, default=None)
    parser.add_argument("--subsets", default=None)
    parser.add_argument("--scenes", default=None)
    parser.add_argument("--camera-ids", default=None)
    skip_group = parser.add_mutually_exclusive_group()
    skip_group.add_argument("--skip-existing", dest="skip_existing", action="store_true")
    skip_group.add_argument("--no-skip-existing", dest="skip_existing", action="store_false")
    progress_group = parser.add_mutually_exclusive_group()
    progress_group.add_argument("--progress", dest="progress", action="store_true")
    progress_group.add_argument("--no-progress", dest="progress", action="store_false")
    parser.set_defaults(progress=True, skip_existing=None)
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()

