"""Package the selected Step 22C V4 candidate and produce reports/figures."""

import argparse
from pathlib import Path

from deep_oc_sort_3d.v4_geometry_refinement.geometry_figures import write_geometry_figures
from deep_oc_sort_3d.v4_geometry_refinement.geometry_package import package_selected_geometry_variant
from deep_oc_sort_3d.v4_geometry_refinement.geometry_refinement_config import load_geometry_refinement_config, progress_default
from deep_oc_sort_3d.v4_geometry_refinement.geometry_report import write_geometry_report
from deep_oc_sort_3d.v4_geometry_refinement.geometry_selector import compare_and_select_geometry_variant


def main() -> None:
    args = _parser().parse_args()
    config = load_geometry_refinement_config(Path(args.config))
    progress = progress_default(config) if args.progress is None else bool(args.progress)
    compare_and_select_geometry_variant(config)
    package = package_selected_geometry_variant(
        config,
        progress=progress,
        overwrite=args.overwrite,
        skip_existing=args.skip_existing,
    )
    report_path = write_geometry_report(config)
    figures = write_geometry_figures(config) if config.get("figures", {}).get("enabled", True) else []
    print("ready: %s" % package.get("ready"))
    print("selected_variant: %s" % package.get("selected_variant"))
    print("rows: %s" % package.get("rows"))
    print("zip_size_mb: %s" % package.get("zip_size_mb"))
    print("zip_path: %s" % package.get("zip_path"))
    print("report: %s" % report_path)
    print("figures_written: %d" % len(figures))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--variant", default=None, help="Accepted for CLI consistency; packaging uses selected_variant.json.")
    parser.add_argument("--all", action="store_true", help="Accepted for CLI consistency.")
    parser.add_argument("--progress", dest="progress", action="store_true")
    parser.add_argument("--no-progress", dest="progress", action="store_false")
    parser.set_defaults(progress=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    return parser


if __name__ == "__main__":
    main()
