"""Audit V5.1 immutable input and train/holdout/val source availability."""

import argparse
import shutil
from pathlib import Path

from deep_oc_sort_3d.v51_geometry_calibration_refit.source_availability_audit import audit_v51_sources
from deep_oc_sort_3d.v51_geometry_calibration_refit.v51_config import load_v51_config, output_root, progress_default
from deep_oc_sort_3d.v51_geometry_calibration_refit.v51_io import read_json


def main() -> None:
    args = _parser().parse_args()
    config = load_v51_config(Path(args.config))
    progress = progress_default(config) if args.progress is None else bool(args.progress)
    directory = output_root(config) / "audit"
    if directory.exists() and args.overwrite:
        shutil.rmtree(str(directory))
    if directory.exists() and args.skip_existing:
        summary = {
            "availability": read_json(directory / "input_availability_audit.json"),
            "camera_mapping": read_json(directory / "camera_mapping_audit.json"),
            "input_summary": read_json(directory / "v4_input_summary.json"),
        }
    else:
        if directory.exists():
            raise FileExistsError("Audit exists; use --overwrite or --skip-existing: %s" % directory)
        summary = audit_v51_sources(config, progress=progress)
    print("input_variant: %s" % summary.get("input_summary", {}).get("input_variant"))
    print("rows: %s" % summary.get("input_summary", {}).get("rows"))
    print("unique_tracks: %s" % summary.get("input_summary", {}).get("unique_tracks"))
    print("fit_train_complete: %s" % summary.get("availability", {}).get("fit_train_complete"))
    print("missing_fit_train_scenes: %s" % summary.get("availability", {}).get("missing_fit_train_scenes"))


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
