"""Select, freeze and zip the valid Step 22B V3.1 candidate."""

import argparse
from pathlib import Path

from deep_oc_sort_3d.v3_coverage_extension.coverage_extension_config import load_coverage_extension_config, output_root, progress_default
from deep_oc_sort_3d.v3_coverage_extension.coverage_extension_metrics import build_comparison_outputs
from deep_oc_sort_3d.v3_coverage_extension.coverage_extension_package import package_selected_candidate
from deep_oc_sort_3d.v3_coverage_extension.coverage_extension_report import write_final_report
from deep_oc_sort_3d.v3_coverage_extension.coverage_extension_selector import select_final_variant


def main() -> None:
    """Recompute selection and package only a passing balanced variant."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--progress", dest="progress", action="store_true")
    parser.add_argument("--no-progress", dest="progress", action="store_false")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--variant", choices=["v3_balanced_coverage_extension"])
    parser.add_argument("--all", action="store_true")
    parser.set_defaults(progress=None)
    args = parser.parse_args()
    del args.variant, args.all
    config = load_coverage_extension_config(args.config)
    progress = progress_default(config) if args.progress is None else bool(args.progress)
    zip_path = output_root(config) / "frozen_candidate" / "packages" / "v3_coverage_extended_official_023_027_track1.zip"
    if zip_path.exists() and args.skip_existing:
        print("Package already exists; skipped: %s" % zip_path)
        return
    comparison = build_comparison_outputs(config, progress=progress)
    selected, verdict = select_final_variant(config, comparison)
    readiness = package_selected_candidate(config, progress=progress, overwrite=args.overwrite)
    write_final_report(config)
    print("selected_variant: %s" % selected)
    print("verdict: %s" % verdict.get("label"))
    print("ready: %s" % readiness.get("v3_coverage_extended_official", {}).get("ready", False))


if __name__ == "__main__":
    main()

