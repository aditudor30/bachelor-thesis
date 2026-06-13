"""Validate Step 22B Track1 variants using official 023-027 rules."""

import argparse
from pathlib import Path

from deep_oc_sort_3d.v3_coverage_extension.coverage_extension_config import VARIANT_NAMES, load_coverage_extension_config, progress_default, variant_root
from deep_oc_sort_3d.v3_coverage_extension.official_track1_validator import validate_variant_track1


def main() -> None:
    """Validate one or all existing coverage-extension variants."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--variant", choices=VARIANT_NAMES)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--progress", dest="progress", action="store_true")
    parser.add_argument("--no-progress", dest="progress", action="store_false")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.set_defaults(progress=None)
    args = parser.parse_args()
    config = load_coverage_extension_config(args.config)
    progress = progress_default(config) if args.progress is None else bool(args.progress)
    variants = list(VARIANT_NAMES) if args.all or args.variant is None else [str(args.variant)]
    for variant in variants:
        root = variant_root(config, variant)
        output = root / "validation_summary.json"
        if output.exists() and args.skip_existing:
            print("Skipped existing validation: %s" % variant)
            continue
        if output.exists() and not args.overwrite:
            raise FileExistsError("Validation output exists; use --overwrite or --skip-existing: %s" % output)
        report = validate_variant_track1(root / "track1_official.txt", output, config, progress=progress)
        print("%s status=%s errors=%s" % (variant, report.get("status"), report.get("num_errors")))


if __name__ == "__main__":
    main()
