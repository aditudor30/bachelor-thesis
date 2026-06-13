"""Audit V5 Track1 input and train/val calibration source availability."""

import argparse
from pathlib import Path

from deep_oc_sort_3d.v5_geometry_calibration.calibration_source_audit import audit_calibration_sources
from deep_oc_sort_3d.v5_geometry_calibration.geometry_calibration_config import load_geometry_calibration_config, output_root, progress_default
from deep_oc_sort_3d.v5_geometry_calibration.geometry_calibration_io import prepare_directory, read_json


def main() -> None:
    args = _parser().parse_args()
    config = load_geometry_calibration_config(Path(args.config))
    progress = progress_default(config) if args.progress is None else bool(args.progress)
    audit_root = output_root(config) / "audit"
    if prepare_directory(audit_root, args.overwrite, args.skip_existing):
        summary = audit_calibration_sources(config, progress=progress)
    else:
        summary = {
            "availability": read_json(audit_root / "input_availability_audit.json"),
            "camera_mapping": read_json(audit_root / "camera_mapping_audit.json"),
            "input_summary": read_json(audit_root / "v4_input_summary.json"),
        }
    print("input_variant: %s" % summary.get("input_summary", {}).get("input_variant"))
    print("rows: %s" % summary.get("input_summary", {}).get("rows"))
    print("unique_tracks: %s" % summary.get("input_summary", {}).get("unique_tracks"))
    print("available_calibration_scenes: %s" % summary.get("availability", {}).get("available_calibration_scenes"))
    print("camera_specific_test_calibration: %s" % summary.get("camera_mapping", {}).get("status"))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    _common(parser)
    return parser


def _common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", required=True)
    parser.add_argument("--variant", default=None)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--progress", dest="progress", action="store_true")
    parser.add_argument("--no-progress", dest="progress", action="store_false")
    parser.set_defaults(progress=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")


if __name__ == "__main__":
    main()
