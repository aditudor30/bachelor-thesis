"""CLI for generic export 3D field audit."""

import argparse
from pathlib import Path
from typing import Any, Dict

from deep_oc_sort_3d.audit3d.audit3d_io import write_csv, write_json
from deep_oc_sort_3d.audit3d.generic_3d_audit import (
    compare_generic_vs_track1,
    compute_generic_3d_stats,
    compute_generic_per_class_stats,
    compute_generic_per_scene_stats,
    read_generic_export_rows,
    stats_dict_to_rows,
)
from deep_oc_sort_3d.audit3d.track1_3d_audit import read_track1_rows


def run(args: Any) -> Dict[str, Any]:
    rows = read_generic_export_rows(args.generic_export_root, show_progress=args.progress)
    summary = compute_generic_3d_stats(rows)
    if args.track1 is not None:
        track1_rows = read_track1_rows(args.track1, show_progress=args.progress)
        summary["generic_vs_track1"] = compare_generic_vs_track1(rows, track1_rows)
    per_class = compute_generic_per_class_stats(rows)
    per_scene = compute_generic_per_scene_stats(rows)
    args.output_root.mkdir(parents=True, exist_ok=True)
    write_json(summary, args.output_root / "generic_3d_field_summary.json")
    write_csv(stats_dict_to_rows(summary), args.output_root / "generic_3d_field_summary.csv")
    write_csv(per_class, args.output_root / "generic_per_class_summary.csv")
    write_csv(per_scene, args.output_root / "generic_per_scene_summary.csv")
    write_json(summary.get("generic_vs_track1", {}), args.output_root / "generic_vs_track1_comparison.json")
    print("Generic rows audited: %d" % len(rows))
    print("Wrote generic 3D audit to %s" % args.output_root)
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit generic export 3D fields.")
    parser.add_argument("--generic-export-root", required=True, type=Path)
    parser.add_argument("--track1", type=Path, default=None)
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
