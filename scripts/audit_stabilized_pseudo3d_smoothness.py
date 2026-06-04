"""Audit smoothness for stabilized pseudo-3D predictions."""

import argparse
from pathlib import Path
from typing import Any, Dict

from deep_oc_sort_3d.audit3d.audit3d_io import iter_data_files, progress_iter, write_csv, write_json
from deep_oc_sort_3d.pseudo3d.pseudo3d_io import read_pseudo3d_predictions_csv
from deep_oc_sort_3d.pseudo3d.pseudo3d_smoothness import audit_pseudo3d_smoothness, worst_pseudo3d_jumps


def run(args: Any) -> Dict[str, Any]:
    predictions = []
    for path in progress_iter(iter_data_files(args.predictions_root, [".csv"]), args.progress, "audit stabilized pseudo3D files", "file"):
        predictions.extend(read_pseudo3d_predictions_csv(path))
    config = {
        "suspicious_step_m": args.suspicious_step_m,
        "invalid_step_m": args.invalid_step_m,
        "suspicious_dimension_cv": args.suspicious_dimension_cv,
        "invalid_dimension_cv": args.invalid_dimension_cv,
        "yaw_jump_threshold": args.yaw_jump_threshold,
    }
    summary = audit_pseudo3d_smoothness(predictions, config)
    per_object = list(summary.get("per_object", []))
    compact = dict(summary)
    compact.pop("per_object", None)
    args.output_root.mkdir(parents=True, exist_ok=True)
    write_json(compact, args.output_root / "stabilized_smoothness_summary.json")
    write_csv(per_object, args.output_root / "stabilized_smoothness_per_track.csv")
    write_csv(worst_pseudo3d_jumps(predictions), args.output_root / "worst_remaining_jumps.csv")
    print("Stabilized smoothness objects: %s" % compact.get("object_count"))
    return compact


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit stabilized pseudo-3D smoothness.")
    parser.add_argument("--predictions-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--suspicious-step-m", type=float, default=3.0)
    parser.add_argument("--invalid-step-m", type=float, default=6.0)
    parser.add_argument("--suspicious-dimension-cv", type=float, default=0.25)
    parser.add_argument("--invalid-dimension-cv", type=float, default=0.50)
    parser.add_argument("--yaw-jump-threshold", type=float, default=1.57)
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
