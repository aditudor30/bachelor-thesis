"""CLI for auditing frozen V3.1 geometry before Step 22C refinement."""

import argparse
from pathlib import Path
from typing import Any, Dict

from deep_oc_sort_3d.v4_geometry_refinement.geometry_audit import audit_v31_geometry
from deep_oc_sort_3d.v4_geometry_refinement.geometry_refinement_config import load_geometry_refinement_config, output_root, progress_default
from deep_oc_sort_3d.v4_geometry_refinement.track1_geometry_io import prepare_directory, read_json


def main() -> None:
    args = _parser().parse_args()
    config = load_geometry_refinement_config(Path(args.config))
    progress = progress_default(config) if args.progress is None else bool(args.progress)
    audit_root = output_root(config) / "audit"
    if not prepare_directory(audit_root, overwrite=args.overwrite, skip_existing=args.skip_existing):
        summary = read_json(audit_root / "v31_geometry_audit.json")
    else:
        summary = audit_v31_geometry(config, progress=progress)
    print("rows: %s" % summary.get("rows"))
    print("unique_tracks: %s" % summary.get("unique_tracks"))
    print("step_p95: %s" % summary.get("step_p95"))
    print("suspect_track_count: %s" % summary.get("suspect_track_count"))
    print("audit_root: %s" % audit_root)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--variant", default=None, help="Accepted for CLI consistency; audit always targets V3.1.")
    parser.add_argument("--all", action="store_true", help="Accepted for CLI consistency.")
    parser.add_argument("--progress", dest="progress", action="store_true")
    parser.add_argument("--no-progress", dest="progress", action="store_false")
    parser.set_defaults(progress=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    return parser


if __name__ == "__main__":
    main()
