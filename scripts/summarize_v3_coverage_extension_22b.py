"""Summarize Step 22B variants and refresh report/figures."""

import argparse
from pathlib import Path

from deep_oc_sort_3d.v3_coverage_extension.coverage_extension_config import load_coverage_extension_config
from deep_oc_sort_3d.v3_coverage_extension.coverage_extension_figures import write_coverage_figures
from deep_oc_sort_3d.v3_coverage_extension.coverage_extension_metrics import build_comparison_outputs
from deep_oc_sort_3d.v3_coverage_extension.coverage_extension_report import write_final_report
from deep_oc_sort_3d.v3_coverage_extension.coverage_extension_selector import select_final_variant


def main() -> None:
    """Read existing artifacts and print the selected candidate summary."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--progress", dest="progress", action="store_true")
    parser.add_argument("--no-progress", dest="progress", action="store_false")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--variant")
    parser.add_argument("--all", action="store_true")
    parser.set_defaults(progress=False)
    args = parser.parse_args()
    del args.overwrite, args.skip_existing, args.variant, args.all
    config = load_coverage_extension_config(args.root / "configs" / "resolved_config.yaml")
    config["v3_coverage_extension"]["output_root"] = str(args.root)
    comparison = build_comparison_outputs(config, progress=bool(args.progress))
    selected, verdict = select_final_variant(config, comparison)
    report = write_final_report(config)
    write_coverage_figures(config)
    candidate = next((row for row in comparison.get("variants", []) if row.get("variant") == selected), None)
    print("verdict: %s" % verdict.get("label"))
    print("selected_variant: %s" % selected)
    if candidate is not None:
        print("track1_rows: %s" % candidate.get("track1_rows"))
        print("unique_tracks: %s" % candidate.get("unique_tracks"))
        print("row_gain_vs_v3: %s" % candidate.get("row_gain_vs_v3"))
        print("validation_errors: %s" % candidate.get("validation_errors"))
    print("report: %s" % report)


if __name__ == "__main__":
    main()
