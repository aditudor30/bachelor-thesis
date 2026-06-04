"""Generate pseudo-3D design documents for Step 15B."""

import argparse
from pathlib import Path
from typing import Any

from deep_oc_sort_3d.audit3d.audit3d_io import write_markdown
from deep_oc_sort_3d.priors3d.pseudo3d_design_report import (
    build_pseudo3d_estimator_design,
    build_pseudo3d_validation_plan,
    build_roadmap_baseline_v2,
)


def run(args: Any) -> None:
    config = {
        "primary_method": args.primary_method,
        "fallback_methods": args.fallback_methods,
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    write_markdown(build_pseudo3d_estimator_design(config), args.output_root / "PSEUDO3D_ESTIMATOR_DESIGN.md")
    write_markdown(build_pseudo3d_validation_plan(), args.output_root / "PSEUDO3D_VALIDATION_PLAN.md")
    write_markdown(build_roadmap_baseline_v2(), args.output_root / "ROADMAP_BASELINE_V2_PSEUDO3D.md")
    print("Wrote pseudo-3D design docs to %s" % args.output_root)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate pseudo-3D design documents.")
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--primary-method", default="bbox_height_depth")
    parser.add_argument("--fallback-methods", nargs="*", default=["bottom_center_ground_approx", "class_default"])
    progress_group = parser.add_mutually_exclusive_group()
    progress_group.add_argument("--progress", dest="progress", action="store_true")
    progress_group.add_argument("--no-progress", dest="progress", action="store_false")
    parser.set_defaults(progress=True)
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
