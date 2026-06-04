"""CLI for Track 1 3D field audit."""

import argparse
from pathlib import Path
from typing import Any, Dict

from deep_oc_sort_3d.audit3d.audit3d_io import write_csv, write_json
from deep_oc_sort_3d.audit3d.track1_3d_audit import (
    compute_3d_field_stats,
    compute_per_class_3d_stats,
    compute_per_scene_3d_stats,
    detect_extreme_3d_values,
    read_track1_rows,
    stats_dict_to_rows,
)


def run(args: Any) -> Dict[str, Any]:
    rows = read_track1_rows(args.track1, show_progress=args.progress)
    config = {
        "coordinate_abs_max": args.coordinate_abs_max,
        "dimension_max": args.dimension_max,
    }
    summary = compute_3d_field_stats(rows)
    per_class = compute_per_class_3d_stats(rows)
    per_scene = compute_per_scene_3d_stats(rows)
    extremes = detect_extreme_3d_values(rows, config)
    args.output_root.mkdir(parents=True, exist_ok=True)
    write_json(summary, args.output_root / "track1_3d_field_summary.json")
    write_csv(stats_dict_to_rows(summary), args.output_root / "track1_3d_field_summary.csv")
    write_csv(per_class, args.output_root / "track1_per_class_summary.csv")
    write_csv(per_scene, args.output_root / "track1_per_scene_summary.csv")
    write_csv(extremes, args.output_root / "track1_extreme_values.csv")
    print("Track1 rows audited: %d" % len(rows))
    print("Wrote Track1 3D audit to %s" % args.output_root)
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit official Track1 3D fields.")
    parser.add_argument("--track1", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--coordinate-abs-max", type=float, default=10000.0)
    parser.add_argument("--dimension-max", type=float, default=20.0)
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
