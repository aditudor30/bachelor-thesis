"""CLI for generating one or all Step 22C V4 geometry variants."""

import argparse
from pathlib import Path
from typing import List

from deep_oc_sort_3d.v4_geometry_refinement.geometry_refinement_config import VARIANT_NAMES, load_geometry_refinement_config, progress_default, variant_root
from deep_oc_sort_3d.v4_geometry_refinement.geometry_variant_runner import run_geometry_variants
from deep_oc_sort_3d.v4_geometry_refinement.track1_geometry_io import prepare_directory


def main() -> None:
    args = _parser().parse_args()
    config = load_geometry_refinement_config(Path(args.config))
    progress = progress_default(config) if args.progress is None else bool(args.progress)
    variants = _variants(args.variant, args.all)
    runnable: List[str] = []
    for variant in variants:
        if prepare_directory(variant_root(config, variant), overwrite=args.overwrite, skip_existing=args.skip_existing):
            runnable.append(variant)
        else:
            print("skip_existing: %s" % variant)
    if not runnable:
        print("No variants required generation.")
        return
    summary = run_geometry_variants(config, runnable, progress=progress)
    for variant in runnable:
        item = summary.get(variant, {})
        print("%s rows=%s changes=%s" % (variant, item.get("rows"), item.get("stage_change_events")))


def _variants(variant: str, all_variants: bool) -> List[str]:
    if variant:
        if variant not in VARIANT_NAMES:
            raise ValueError("Unknown variant: %s" % variant)
        return [variant]
    if all_variants:
        return list(VARIANT_NAMES)
    return list(VARIANT_NAMES)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--variant", choices=VARIANT_NAMES)
    parser.add_argument("--all", action="store_true", help="Generate all five V4 variants.")
    parser.add_argument("--progress", dest="progress", action="store_true")
    parser.add_argument("--no-progress", dest="progress", action="store_false")
    parser.set_defaults(progress=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    return parser


if __name__ == "__main__":
    main()
