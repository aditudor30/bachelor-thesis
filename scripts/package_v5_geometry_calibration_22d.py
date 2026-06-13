"""Freeze selected V5, create upload ZIP, report and figures."""

import argparse
from pathlib import Path

from deep_oc_sort_3d.v5_geometry_calibration.correction_selector import compare_and_select_v5_variant
from deep_oc_sort_3d.v5_geometry_calibration.geometry_calibration_config import load_geometry_calibration_config, progress_default
from deep_oc_sort_3d.v5_geometry_calibration.geometry_calibration_figures import write_v5_figures
from deep_oc_sort_3d.v5_geometry_calibration.geometry_calibration_package import package_selected_v5
from deep_oc_sort_3d.v5_geometry_calibration.geometry_calibration_report import write_v5_report


def main() -> None:
    args = _parser().parse_args()
    config = load_geometry_calibration_config(Path(args.config))
    progress = progress_default(config) if args.progress is None else bool(args.progress)
    compare_and_select_v5_variant(config)
    package = package_selected_v5(config, progress=progress, overwrite=args.overwrite, skip_existing=args.skip_existing)
    report = write_v5_report(config)
    figures = write_v5_figures(config) if config.get("figures", {}).get("enabled", True) else []
    print("ready: %s" % package.get("ready"))
    print("selected_variant: %s" % package.get("selected_variant"))
    print("rows: %s" % package.get("rows"))
    print("unique_tracks: %s" % package.get("unique_tracks"))
    print("zip_path: %s" % package.get("zip_path"))
    print("report: %s" % report)
    print("figures_written: %d" % len(figures))


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
