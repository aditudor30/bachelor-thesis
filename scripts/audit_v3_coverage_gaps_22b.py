"""CLI for Step 22B V3/V2 coverage-gap and source audit."""

import argparse
from pathlib import Path

from deep_oc_sort_3d.v3_coverage_extension.coverage_extension_config import load_coverage_extension_config, progress_default
from deep_oc_sort_3d.v3_coverage_extension.coverage_extension_io import prepare_directory
from deep_oc_sort_3d.v3_coverage_extension.coverage_gap_audit import run_coverage_gap_audit


def main() -> None:
    """Run the isolated Step 22B audit."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--progress", dest="progress", action="store_true")
    parser.add_argument("--no-progress", dest="progress", action="store_false")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--variant")
    parser.add_argument("--all", action="store_true")
    parser.set_defaults(progress=None)
    args = parser.parse_args()
    del args.variant, args.all
    config = load_coverage_extension_config(args.config)
    progress = progress_default(config) if args.progress is None else bool(args.progress)
    root = Path(str(config.get("v3_coverage_extension", {}).get("output_root")))
    if not prepare_directory(root / "audit", args.overwrite, args.skip_existing):
        print("Audit already exists; skipped.")
        return
    summary = run_coverage_gap_audit(config, progress=progress)
    print("V2 rows: %s" % summary.get("v2", {}).get("rows"))
    print("V3 rows: %s" % summary.get("v3", {}).get("rows"))
    print("Recovery local tracks: %s" % summary.get("sources", {}).get("local_tracks"))
    print("Uncovered local tracks: %s" % summary.get("sources", {}).get("uncovered_local_tracks"))


if __name__ == "__main__":
    main()
