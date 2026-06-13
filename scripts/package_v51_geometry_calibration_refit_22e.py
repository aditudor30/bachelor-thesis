"""Package the selected V5.1 candidate and write figures plus report."""

import argparse
from pathlib import Path

from deep_oc_sort_3d.v51_geometry_calibration_refit.correction_selector import compare_and_select_v51_variant
from deep_oc_sort_3d.v51_geometry_calibration_refit.v51_config import load_v51_config, progress_default
from deep_oc_sort_3d.v51_geometry_calibration_refit.v51_figures import write_v51_figures
from deep_oc_sort_3d.v51_geometry_calibration_refit.v51_package import package_selected_v51
from deep_oc_sort_3d.v51_geometry_calibration_refit.v51_report import write_v51_report


def main() -> None:
    args = _parser().parse_args()
    config = load_v51_config(Path(args.config))
    progress = progress_default(config) if args.progress is None else bool(args.progress)
    compare_and_select_v51_variant(config)
    candidate = package_selected_v51(config, progress=progress, overwrite=args.overwrite, skip_existing=args.skip_existing)
    if bool(config.get("figures", {}).get("enabled", True)):
        write_v51_figures(config)
    report = write_v51_report(config)
    print("ready: %s" % candidate.get("ready"))
    print("track1: %s" % candidate.get("track1_path"))
    print("zip: %s" % candidate.get("zip_path"))
    print("report: %s" % report)


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
