"""Run the complete Step 22E refit workflow or selected V5.1 variants."""

import argparse
import shutil
from pathlib import Path
from typing import List

from deep_oc_sort_3d.v51_geometry_calibration_refit.calibration_dataset_builder import build_v51_calibration_dataset
from deep_oc_sort_3d.v51_geometry_calibration_refit.correction_selector import compare_and_select_v51_variant, fit_and_select_corrections
from deep_oc_sort_3d.v51_geometry_calibration_refit.source_availability_audit import audit_v51_sources
from deep_oc_sort_3d.v51_geometry_calibration_refit.track1_validator import validate_v51_track1
from deep_oc_sort_3d.v51_geometry_calibration_refit.v51_config import VARIANT_NAMES, load_v51_config, output_root, progress_default, variant_root
from deep_oc_sort_3d.v51_geometry_calibration_refit.v51_figures import write_v51_figures
from deep_oc_sort_3d.v51_geometry_calibration_refit.v51_io import read_json
from deep_oc_sort_3d.v51_geometry_calibration_refit.v51_package import package_selected_v51
from deep_oc_sort_3d.v51_geometry_calibration_refit.v51_report import write_v51_report
from deep_oc_sort_3d.v51_geometry_calibration_refit.v51_variant_runner import run_v51_variants


def main() -> None:
    args = _parser().parse_args()
    config = load_v51_config(Path(args.config))
    progress = progress_default(config) if args.progress is None else bool(args.progress)
    variants = [args.variant] if args.variant else list(VARIANT_NAMES)
    root = output_root(config)
    if args.all:
        if not (root / "audit" / "input_availability_audit.json").is_file():
            audit_v51_sources(config, progress=progress)
        if not (root / "calibration_dataset" / "match_rate_summary.json").is_file():
            build_v51_calibration_dataset(config, progress=progress, overwrite=args.overwrite)
    if args.overwrite:
        for directory in [root / "learned_corrections", root / "validation_diagnostics", root / "comparison"]:
            if directory.exists():
                shutil.rmtree(str(directory))
    verdict_path = root / "validation_diagnostics" / "calibration_verdict.json"
    selected_path = root / "learned_corrections" / "selected_corrections.json"
    if args.skip_existing and verdict_path.is_file() and selected_path.is_file():
        verdict = read_json(verdict_path)
    else:
        verdict = fit_and_select_corrections(config)
    runnable: List[str] = []
    for variant in variants:
        directory = variant_root(config, variant)
        if directory.exists() and args.skip_existing:
            continue
        if directory.exists() and args.overwrite:
            shutil.rmtree(str(directory))
        elif directory.exists():
            raise FileExistsError("Variant exists; use --overwrite or --skip-existing: %s" % directory)
        directory.mkdir(parents=True, exist_ok=True)
        runnable.append(variant)
    if runnable:
        run_v51_variants(config, runnable, progress=progress)
    for variant in variants:
        directory = variant_root(config, variant)
        if (directory / "track1.txt").is_file():
            validate_v51_track1(directory / "track1.txt", directory / "validation_summary.json", config, progress=progress)
    comparison = compare_and_select_v51_variant(config, variants=variants)
    if args.all:
        package_selected_v51(config, progress=progress, overwrite=args.overwrite, skip_existing=args.skip_existing)
        if bool(config.get("figures", {}).get("enabled", True)):
            write_v51_figures(config)
        write_v51_report(config)
    print("fit_source: %s" % verdict.get("fit_source"))
    print("selected_components: %s" % verdict.get("selected_components"))
    print("selected_variant: %s" % comparison.get("selected_variant"))
    print("verdict: %s" % comparison.get("verdict"))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--variant", choices=VARIANT_NAMES)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--progress", dest="progress", action="store_true")
    parser.add_argument("--no-progress", dest="progress", action="store_false")
    parser.set_defaults(progress=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    return parser


if __name__ == "__main__":
    main()
