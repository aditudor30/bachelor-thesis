"""Compare baseline_v1_geometry_only and baseline_v2_pseudo3d_fullcam."""

import argparse
from pathlib import Path

from deep_oc_sort_3d.baseline_v2_fullcam.fullcam_comparison import (
    compare_fullcam_from_config,
    write_fullcam_comparison_outputs,
)
from deep_oc_sort_3d.baseline_v2_fullcam.fullcam_observation_builder import load_fullcam_config
from deep_oc_sort_3d.baseline_v2_fullcam.fullcam_report import write_fullcam_report


def build_arg_parser() -> argparse.ArgumentParser:
    """Build CLI parser."""
    parser = argparse.ArgumentParser(description="Compare baseline V1 and V2 fullcam pseudo3D outputs.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    progress_group = parser.add_mutually_exclusive_group()
    progress_group.add_argument("--progress", dest="progress", action="store_true")
    progress_group.add_argument("--no-progress", dest="progress", action="store_false")
    parser.set_defaults(progress=True)
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    config = load_fullcam_config(args.config)
    output_root = Path(config.get("paths", {}).get("output_comparison_root", "output/baseline_v2_pseudo3d_fullcam_comparison"))
    summary = compare_fullcam_from_config(config)
    write_fullcam_comparison_outputs(summary, output_root)
    write_fullcam_report(summary, output_root / "BASELINE_V1_VS_V2_PSEUDO3D_FULLCAM_REPORT.md")
    print("verdict: %s" % summary.get("verdict", {}).get("label"))
    print("Wrote %s" % output_root)


if __name__ == "__main__":
    main()
