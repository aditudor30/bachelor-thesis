"""Validate V5 variants and select a train/val-justified safe candidate."""

import argparse
from pathlib import Path

from deep_oc_sort_3d.v5_geometry_calibration.correction_selector import compare_and_select_v5_variant
from deep_oc_sort_3d.v5_geometry_calibration.geometry_calibration_config import VARIANT_NAMES, load_geometry_calibration_config, progress_default, variant_root
from deep_oc_sort_3d.v5_geometry_calibration.geometry_calibration_io import read_json
from deep_oc_sort_3d.v5_geometry_calibration.track1_validator import validate_v5_track1


def main() -> None:
    args = _parser().parse_args()
    config = load_geometry_calibration_config(Path(args.config))
    progress = progress_default(config) if args.progress is None else bool(args.progress)
    variants = [args.variant] if args.variant else list(VARIANT_NAMES)
    for variant in variants:
        root = variant_root(config, variant)
        report_path = root / "validation_summary.json"
        if report_path.exists() and args.skip_existing:
            report = read_json(report_path)
        else:
            if report_path.exists() and not args.overwrite:
                raise FileExistsError("Validation exists; use --overwrite or --skip-existing: %s" % report_path)
            report = validate_v5_track1(root / "track1.txt", report_path, config, progress=progress)
        print("%s status=%s errors=%s" % (variant, report.get("status"), report.get("num_errors")))
    comparison = compare_and_select_v5_variant(config, variants=variants)
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
