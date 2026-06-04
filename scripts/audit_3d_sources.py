"""CLI for 3D source metadata audit."""

import argparse
from pathlib import Path
from typing import Any, Dict

from deep_oc_sort_3d.audit3d.audit3d_io import write_csv, write_json
from deep_oc_sort_3d.audit3d.source_3d_audit import audit_3d_sources, write_missing_source_metadata_report


def run(args: Any) -> Dict[str, Any]:
    summary = audit_3d_sources(
        args.frame_records_root,
        observations_root=args.observations_root,
        candidates_root=args.candidates_root,
        show_progress=args.progress,
    )
    args.output_root.mkdir(parents=True, exist_ok=True)
    write_json(summary, args.output_root / "source_3d_summary.json")
    write_csv(list(summary.get("per_subset", [])), args.output_root / "source_3d_per_subset.csv")
    write_missing_source_metadata_report(summary, args.output_root / "missing_source_metadata_report.md")
    print("Records audited: %s" % summary.get("record_count"))
    print("Wrote source audit to %s" % args.output_root)
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit 3D source metadata.")
    parser.add_argument("--frame-records-root", required=True, type=Path)
    parser.add_argument("--observations-root", type=Path, default=None)
    parser.add_argument("--candidates-root", type=Path, default=None)
    parser.add_argument("--output-root", required=True, type=Path)
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

