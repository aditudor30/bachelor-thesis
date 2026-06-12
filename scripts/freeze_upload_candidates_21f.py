"""Freeze, validate, package and compare the two Step 21F candidates."""

import argparse
from pathlib import Path

from deep_oc_sort_3d.freeze_upload.freeze_comparison import compare_frozen_candidates
from deep_oc_sort_3d.freeze_upload.freeze_config import load_freeze_config
from deep_oc_sort_3d.freeze_upload.freeze_manager import freeze_candidates
from deep_oc_sort_3d.freeze_upload.freeze_report import write_freeze_report
from deep_oc_sort_3d.freeze_upload.package_builder import package_frozen_candidates


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze Step 21F upload candidates")
    parser.add_argument("--config", type=Path, required=True)
    _add_common_options(parser, include_no_zip=True)
    args = parser.parse_args()
    config = load_freeze_config(args.config)
    freeze_candidates(config, progress=args.progress, overwrite=args.overwrite, skip_existing=args.skip_existing)
    package_frozen_candidates(
        config,
        progress=args.progress,
        overwrite=args.overwrite,
        skip_existing=args.skip_existing,
        create_zip=bool(config.get("freeze_upload_candidates", {}).get("create_zip_packages", True)) and not args.no_zip,
    )
    comparison = compare_frozen_candidates(config, progress=args.progress)
    report_path = write_freeze_report(config, comparison)
    print("verdict: %s" % comparison.get("verdict", {}).get("label"))
    print("report: %s" % report_path)


def _add_common_options(parser: argparse.ArgumentParser, include_no_zip: bool = False) -> None:
    progress_group = parser.add_mutually_exclusive_group()
    progress_group.add_argument("--progress", dest="progress", action="store_true", default=True)
    progress_group.add_argument("--no-progress", dest="progress", action="store_false")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    if include_no_zip:
        parser.add_argument("--no-zip", action="store_true")


if __name__ == "__main__":
    main()
