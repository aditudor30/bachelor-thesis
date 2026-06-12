"""Run one Step 21B stage."""

import argparse
from pathlib import Path

from deep_oc_sort_3d.bytetrack_local_pipeline.bytetrack_local_runner import run_bytetrack_local_tracking
from deep_oc_sort_3d.bytetrack_local_pipeline.bytetrack_pipeline_config import (
    load_bytetrack_pipeline_config,
    selected_stages,
)
from deep_oc_sort_3d.bytetrack_local_pipeline.bytetrack_precheck import run_bytetrack_precheck
from deep_oc_sort_3d.bytetrack_local_pipeline.bytetrack_precheck import precheck_allows_full_rerun
from deep_oc_sort_3d.bytetrack_local_pipeline.bytetrack_io import read_json
from deep_oc_sort_3d.bytetrack_local_pipeline.bytetrack_stage_runner import run_bytetrack_stages


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run one baseline V2 ByteTrack-local stage")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--stage", required=True, choices=selected_stages(None))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--progress", dest="progress", action="store_true")
    group.add_argument("--no-progress", dest="progress", action="store_false")
    parser.set_defaults(progress=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.stage == "precheck":
        run_bytetrack_precheck(args.config, progress=args.progress, overwrite=args.overwrite)
        return
    if args.stage == "local_tracking":
        config = load_bytetrack_pipeline_config(args.config)
        if bool(config.get("precheck", {}).get("require_pass_before_full_rerun", True)):
            verdict_path = Path(str(config.get("paths", {}).get("output_precheck_root"))) / "precheck_verdict.json"
            if not precheck_allows_full_rerun(read_json(verdict_path)):
                raise RuntimeError("ByteTrack local tracking blocked because precheck has not passed")
        summary = run_bytetrack_local_tracking(config, progress=args.progress, overwrite=args.overwrite)
        print("records: %s" % summary.get("num_track_records"))
        print("errors: %s" % summary.get("errors"))
        return
    run_bytetrack_stages(
        args.config,
        stages=[args.stage],
        progress=args.progress,
        overwrite=args.overwrite,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
