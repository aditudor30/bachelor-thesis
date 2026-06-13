"""Validate V4 variants, compare them with V3.1 and select conservatively."""

import argparse
from pathlib import Path
from typing import List

from deep_oc_sort_3d.v4_geometry_refinement.geometry_refinement_config import VARIANT_NAMES, load_geometry_refinement_config, progress_default, variant_root
from deep_oc_sort_3d.v4_geometry_refinement.geometry_selector import compare_and_select_geometry_variant
from deep_oc_sort_3d.v4_geometry_refinement.track1_geometry_io import read_json
from deep_oc_sort_3d.v4_geometry_refinement.track1_geometry_validator import validate_v4_track1


def main() -> None:
    args = _parser().parse_args()
    config = load_geometry_refinement_config(Path(args.config))
    progress = progress_default(config) if args.progress is None else bool(args.progress)
    variants = [args.variant] if args.variant else list(VARIANT_NAMES)
    for variant in variants:
        root = variant_root(config, variant)
        report_path = root / "validation_summary.json"
        if args.skip_existing and report_path.is_file():
            report = read_json(report_path)
        else:
            if report_path.exists() and not args.overwrite:
                raise FileExistsError("Validation exists; use --overwrite or --skip-existing: %s" % report_path)
            report = validate_v4_track1(root / "track1.txt", report_path, config, progress=progress)
        geometry_path = root / "geometry_summary.json"
        geometry = read_json(geometry_path)
        checks = report.get("checks", {})
        geometry.update({
            "validation_errors": report.get("num_errors"),
            "duplicate_keys": checks.get("duplicate_key_count"),
            "nan_inf": checks.get("nan_or_inf_values"),
            "non_positive_dimensions": checks.get("non_positive_dimensions"),
            "rounding_issues": checks.get("rounding_issues"),
        })
        from deep_oc_sort_3d.v4_geometry_refinement.track1_geometry_io import write_json
        write_json(geometry_path, geometry)
        print("%s status=%s errors=%s" % (variant, report.get("status"), report.get("num_errors")))
    comparison = compare_and_select_geometry_variant(config, variants=VARIANT_NAMES)
    print("selected_variant: %s" % comparison.get("selected_variant"))
    print("verdict: %s" % comparison.get("verdict"))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--variant", choices=VARIANT_NAMES)
    parser.add_argument("--all", action="store_true", help="Validate every generated variant.")
    parser.add_argument("--progress", dest="progress", action="store_true")
    parser.add_argument("--no-progress", dest="progress", action="store_false")
    parser.set_defaults(progress=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    return parser


if __name__ == "__main__":
    main()
