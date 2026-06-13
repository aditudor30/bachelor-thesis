"""Build controlled V3.1 official coverage-extension variants."""

import argparse
from pathlib import Path
from typing import Any, Dict, List

from deep_oc_sort_3d.v3_coverage_extension.balanced_extension_builder import run_variant_builds
from deep_oc_sort_3d.v3_coverage_extension.coverage_extension_config import VARIANT_NAMES, load_coverage_extension_config, output_root, progress_default, variant_root
from deep_oc_sort_3d.v3_coverage_extension.coverage_extension_figures import write_coverage_figures
from deep_oc_sort_3d.v3_coverage_extension.coverage_extension_io import prepare_directory
from deep_oc_sort_3d.v3_coverage_extension.coverage_extension_metrics import build_comparison_outputs
from deep_oc_sort_3d.v3_coverage_extension.coverage_extension_package import package_selected_candidate
from deep_oc_sort_3d.v3_coverage_extension.coverage_extension_report import write_final_report
from deep_oc_sort_3d.v3_coverage_extension.coverage_extension_selector import select_final_variant
from deep_oc_sort_3d.v3_coverage_extension.coverage_gap_audit import run_coverage_gap_audit
from deep_oc_sort_3d.v3_coverage_extension.official_track1_validator import validate_variant_track1


def main() -> None:
    """Build one variant or execute the full Step 22B workflow."""
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
    requested = list(VARIANT_NAMES) if args.all or args.variant is None else [str(args.variant)]
    if args.all:
        prepare_directory(output_root(config) / "audit", args.overwrite, False)
        run_coverage_gap_audit(config, progress=progress)
    runnable = []
    for variant in requested:
        if prepare_directory(variant_root(config, variant), args.overwrite, args.skip_existing):
            runnable.append(variant)
        else:
            print("Skipped existing variant: %s" % variant)
    result = run_variant_builds(config, runnable, progress=progress) if runnable else {"variants": {}}
    for variant, item in result.get("variants", {}).items():
        print("%s rows=%s added=%s" % (variant, item.get("manifest", {}).get("output_rows"), item.get("added_rows_summary", {}).get("added_rows")))
    if args.all:
        _validate_variants(config, requested, progress)
        comparison = build_comparison_outputs(config, progress=progress)
        select_final_variant(config, comparison)
        package_selected_candidate(config, progress=progress, overwrite=args.overwrite)
        write_final_report(config)
        write_coverage_figures(config)
        print("Step 22B output root: %s" % output_root(config))


def _validate_variants(config: Dict[str, Any], variants: List[str], progress: bool) -> None:
    for variant in variants:
        root = variant_root(config, variant)
        path = root / "track1_official.txt"
        if path.exists():
            report = validate_variant_track1(path, root / "validation_summary.json", config, progress=progress)
            print("%s validation=%s errors=%s" % (variant, report.get("status"), report.get("num_errors")))


if __name__ == "__main__":
    main()

