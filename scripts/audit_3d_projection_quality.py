"""CLI for 3D cuboid projection audit."""

import argparse
from pathlib import Path
from typing import Any, Dict

from deep_oc_sort_3d.audit3d.audit3d_io import write_csv, write_json
from deep_oc_sort_3d.audit3d.projection_3d_audit import (
    audit_projection_for_records,
    projection_failures_to_rows,
)


def run(args: Any) -> Dict[str, Any]:
    summary = audit_projection_for_records(
        args.dataset_root,
        args.records_csv,
        split=args.split,
        scene_name=args.scene,
        camera_id=args.camera_id,
        max_records=args.max_records,
        show_progress=args.progress,
    )
    args.output_root.mkdir(parents=True, exist_ok=True)
    write_json(summary, args.output_root / "projection_audit_summary.json")
    write_csv(projection_failures_to_rows({"items": [summary]}), args.output_root / "projection_failures.csv")
    print("Projection success rate: %s" % summary.get("success_rate"))
    print("Wrote projection audit to %s" % args.output_root)
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit 3D projection quality for one records CSV/camera.")
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--records-csv", required=True, type=Path)
    parser.add_argument("--split", required=True)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--camera-id", required=True)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--max-records", type=int, default=500)
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

