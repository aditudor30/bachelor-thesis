"""Fit selected train/val corrections and generate V5 Track1 variants."""

import argparse
import shutil
from pathlib import Path
from typing import List

from deep_oc_sort_3d.v5_geometry_calibration.correction_selector import fit_and_select_corrections
from deep_oc_sort_3d.v5_geometry_calibration.geometry_calibration_config import VARIANT_NAMES, load_geometry_calibration_config, output_root, progress_default, variant_root
from deep_oc_sort_3d.v5_geometry_calibration.geometry_calibration_io import read_json
from deep_oc_sort_3d.v5_geometry_calibration.geometry_calibration_variant_runner import run_calibration_variants


def main() -> None:
    args = _parser().parse_args()
    config = load_geometry_calibration_config(Path(args.config))
    progress = progress_default(config) if args.progress is None else bool(args.progress)
    variants = [args.variant] if args.variant else list(VARIANT_NAMES)
    root = output_root(config)
    fit_directories = [root / "learned_corrections", root / "validation_diagnostics"]
    for directory in fit_directories:
        if directory.exists() and args.overwrite:
            shutil.rmtree(str(directory))
        elif directory.exists() and not args.skip_existing:
            raise FileExistsError("Output exists; use --overwrite or --skip-existing: %s" % directory)
    selected_path = root / "learned_corrections" / "selected_corrections.json"
    verdict_path = root / "validation_diagnostics" / "calibration_verdict.json"
    fit_outputs_ready = selected_path.is_file() and verdict_path.is_file()
    if args.skip_existing and any(directory.exists() for directory in fit_directories) and not fit_outputs_ready:
        raise FileNotFoundError("Partial calibration outputs found; use --overwrite to rebuild them")
    verdict = read_json(verdict_path) if args.skip_existing and fit_outputs_ready else fit_and_select_corrections(config)
    runnable: List[str] = []
    for variant in variants:
        directory = variant_root(config, variant)
        if directory.exists() and args.skip_existing:
            print("skip_existing: %s" % variant)
            continue
        if directory.exists() and not args.overwrite:
            raise FileExistsError("Variant exists; use --overwrite or --skip-existing: %s" % directory)
        if directory.exists():
            shutil.rmtree(str(directory))
        directory.mkdir(parents=True, exist_ok=True)
        runnable.append(variant)
    result = run_calibration_variants(config, runnable, progress=progress) if runnable else {}
    print("fit_source: %s" % verdict.get("fit_source"))
    print("selected_components: %s" % verdict.get("selected_components"))
    for variant in runnable:
        metrics = result.get(variant, {}).get("geometry_summary", {})
        print("%s rows=%s tracks=%s" % (variant, metrics.get("rows"), metrics.get("unique_tracks")))


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
