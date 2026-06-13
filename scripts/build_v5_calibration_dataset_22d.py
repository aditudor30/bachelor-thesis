"""Build conservative pseudo3D-to-GT matches using train/val only."""

import argparse
from pathlib import Path

from deep_oc_sort_3d.v5_geometry_calibration.calibration_dataset_builder import build_calibration_dataset
from deep_oc_sort_3d.v5_geometry_calibration.geometry_calibration_config import load_geometry_calibration_config, output_root, progress_default
from deep_oc_sort_3d.v5_geometry_calibration.geometry_calibration_io import prepare_directory, read_json


def main() -> None:
    args = _parser().parse_args()
    config = load_geometry_calibration_config(Path(args.config))
    progress = progress_default(config) if args.progress is None else bool(args.progress)
    directory = output_root(config) / "calibration_dataset"
    if prepare_directory(directory, args.overwrite, args.skip_existing):
        summary = build_calibration_dataset(config, progress=progress)
    else:
        summary = read_json(directory / "calibration_matches_summary.json")
    print("num_predictions: %s" % summary.get("num_predictions"))
    print("num_gt: %s" % summary.get("num_gt"))
    print("num_matches: %s" % summary.get("num_matches"))
    print("match_rate: %s" % summary.get("match_rate"))
    print("samples_per_phase: %s" % summary.get("samples_per_phase"))
    print("samples_per_class: %s" % summary.get("samples_per_class"))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--variant", default=None)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--progress", dest="progress", action="store_true")
    parser.add_argument("--no-progress", dest="progress", action="store_false")
    parser.set_defaults(progress=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    return parser


if __name__ == "__main__":
    main()
