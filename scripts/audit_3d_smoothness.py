"""CLI for Track1 3D smoothness audit."""

import argparse
from pathlib import Path
from typing import Any, Dict

from deep_oc_sort_3d.audit3d.audit3d_io import write_csv, write_json
from deep_oc_sort_3d.audit3d.smoothness_3d_audit import (
    compute_smoothness_audit,
    find_worst_dimension_variation,
    find_worst_jumps,
)
from deep_oc_sort_3d.audit3d.track1_3d_audit import read_track1_rows


def run(args: Any) -> Dict[str, Any]:
    rows = read_track1_rows(args.track1, show_progress=args.progress)
    config = {
        "suspicious_step_m": args.suspicious_step_m,
        "invalid_step_m": args.invalid_step_m,
        "suspicious_dimension_cv": args.suspicious_dimension_cv,
        "invalid_dimension_cv": args.invalid_dimension_cv,
        "yaw_jump_threshold": args.yaw_jump_threshold,
    }
    audit = compute_smoothness_audit(rows, config, show_progress=args.progress)
    per_object = list(audit.get("per_object", []))
    summary = dict(audit)
    summary.pop("per_object", None)
    args.output_root.mkdir(parents=True, exist_ok=True)
    write_json(summary, args.output_root / "track_smoothness_summary.json")
    write_csv(per_object, args.output_root / "track_smoothness_per_object.csv")
    write_csv(find_worst_jumps(rows, top_k=args.top_k), args.output_root / "worst_3d_jumps.csv")
    write_csv(find_worst_dimension_variation(rows, top_k=args.top_k), args.output_root / "worst_dimension_variation.csv")
    print("Objects audited: %s" % summary.get("object_count"))
    print("Wrote smoothness audit to %s" % args.output_root)
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit Track1 3D trajectory smoothness.")
    parser.add_argument("--track1", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--suspicious-step-m", type=float, default=3.0)
    parser.add_argument("--invalid-step-m", type=float, default=6.0)
    parser.add_argument("--suspicious-dimension-cv", type=float, default=0.25)
    parser.add_argument("--invalid-dimension-cv", type=float, default=0.50)
    parser.add_argument("--yaw-jump-threshold", type=float, default=1.57)
    parser.add_argument("--top-k", type=int, default=100)
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
