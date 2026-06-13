"""Build leakage-free V5.1 calibration matches for all three splits."""

import argparse
import shutil
from pathlib import Path

from deep_oc_sort_3d.v51_geometry_calibration_refit.calibration_dataset_builder import build_v51_calibration_dataset
from deep_oc_sort_3d.v51_geometry_calibration_refit.v51_config import load_v51_config, output_root, progress_default
from deep_oc_sort_3d.v51_geometry_calibration_refit.v51_io import read_json


def main() -> None:
    args = _parser().parse_args()
    config = load_v51_config(Path(args.config))
    progress = progress_default(config) if args.progress is None else bool(args.progress)
    directory = output_root(config) / "calibration_dataset"
    if directory.exists() and args.overwrite:
        shutil.rmtree(str(directory))
    if directory.exists() and args.skip_existing:
        summary = read_json(directory / "match_rate_summary.json")
    else:
        if directory.exists():
            raise FileExistsError("Calibration dataset exists; use --overwrite or --skip-existing: %s" % directory)
        summary = build_v51_calibration_dataset(config, progress=progress, overwrite=args.overwrite)
    for phase in ["fit_train", "internal_holdout", "official_val"]:
        item = summary.get(phase, {})
        print("%s matches=%s match_rate=%s" % (phase, item.get("num_matches"), item.get("match_rate")))


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
