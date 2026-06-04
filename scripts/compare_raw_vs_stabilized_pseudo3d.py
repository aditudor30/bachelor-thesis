"""Compare raw and stabilized pseudo-3D predictions."""

import argparse
from pathlib import Path
from typing import Any, Dict, List

from deep_oc_sort_3d.audit3d.audit3d_io import iter_data_files, progress_iter, write_csv, write_json
from deep_oc_sort_3d.pseudo3d.pseudo3d_io import flatten_prediction_dict
from deep_oc_sort_3d.pseudo3d.pseudo3d_smoothness import audit_pseudo3d_smoothness, worst_pseudo3d_jumps
from deep_oc_sort_3d.pseudo3d.pseudo3d_stabilization_eval import compare_raw_and_stabilized, comparison_rows
from deep_oc_sort_3d.pseudo3d.pseudo3d_stabilization_io import outputs_to_dicts, read_pseudo3d_outputs


def run(args: Any) -> Dict[str, Any]:
    raw_outputs = _read_outputs(args.raw_root, args.progress, "read raw pseudo3D files")
    stabilized_outputs = _read_outputs(args.stabilized_root, args.progress, "read stabilized pseudo3D files")
    comparison = compare_raw_and_stabilized(raw_outputs, stabilized_outputs)
    raw_rows = _flatten_outputs(raw_outputs)
    stabilized_rows = _flatten_outputs(stabilized_outputs)
    thresholds = {
        "suspicious_step_m": args.suspicious_step_m,
        "invalid_step_m": args.invalid_step_m,
        "suspicious_dimension_cv": args.suspicious_dimension_cv,
        "invalid_dimension_cv": args.invalid_dimension_cv,
        "yaw_jump_threshold": args.yaw_jump_threshold,
    }
    raw_smoothness = audit_pseudo3d_smoothness(raw_rows, thresholds)
    stabilized_smoothness = audit_pseudo3d_smoothness(stabilized_rows, thresholds)
    raw_compact = _compact_smoothness(raw_smoothness)
    stabilized_compact = _compact_smoothness(stabilized_smoothness)
    summary = {
        "comparison": comparison,
        "raw_smoothness": raw_compact,
        "stabilized_smoothness": stabilized_compact,
        "smoothness_delta": _smoothness_delta(raw_compact, stabilized_compact),
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    write_json(summary, args.output_root / "raw_vs_stabilized_smoothness.json")
    write_csv(_comparison_csv_rows(summary), args.output_root / "raw_vs_stabilized_smoothness.csv")
    write_csv(list(stabilized_smoothness.get("per_object", [])), args.output_root / "stabilized_smoothness_per_track.csv")
    write_csv(worst_pseudo3d_jumps(stabilized_rows), args.output_root / "worst_remaining_jumps.csv")
    print("Raw invalid tracks: %s" % raw_compact.get("status_distribution", {}).get("invalid"))
    print("Stabilized invalid tracks: %s" % stabilized_compact.get("status_distribution", {}).get("invalid"))
    return summary


def _read_outputs(root: Path, progress: bool, desc: str) -> List[Any]:
    outputs = []
    files = [path for path in iter_data_files(root, [".jsonl"]) if "stabilized" in path.name or "predictions" in path.name]
    for path in progress_iter(files, progress, desc, "file"):
        outputs.extend(read_pseudo3d_outputs(path))
    return outputs


def _flatten_outputs(outputs: List[Any]) -> List[Dict[str, Any]]:
    return [flatten_prediction_dict(row) for row in outputs_to_dicts(outputs)]


def _compact_smoothness(summary: Dict[str, Any]) -> Dict[str, Any]:
    compact = dict(summary)
    compact.pop("per_object", None)
    return compact


def _smoothness_delta(raw: Dict[str, Any], stabilized: Dict[str, Any]) -> Dict[str, Any]:
    raw_invalid = _status_count(raw, "invalid")
    stabilized_invalid = _status_count(stabilized, "invalid")
    raw_objects = int(raw.get("object_count", 0) or 0)
    stabilized_objects = int(stabilized.get("object_count", 0) or 0)
    return {
        "invalid_count_delta": stabilized_invalid - raw_invalid,
        "raw_invalid_rate": float(raw_invalid) / float(raw_objects) if raw_objects else None,
        "stabilized_invalid_rate": float(stabilized_invalid) / float(stabilized_objects) if stabilized_objects else None,
        "raw_step_p95": raw.get("step_distance_max_stats", {}).get("p95"),
        "stabilized_step_p95": stabilized.get("step_distance_max_stats", {}).get("p95"),
        "raw_step_max": raw.get("step_distance_max_stats", {}).get("max"),
        "stabilized_step_max": stabilized.get("step_distance_max_stats", {}).get("max"),
    }


def _status_count(summary: Dict[str, Any], status: str) -> int:
    return int(summary.get("status_distribution", {}).get(status, 0) or 0)


def _comparison_csv_rows(summary: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows = comparison_rows(summary.get("comparison", {}))
    delta = summary.get("smoothness_delta", {})
    rows.append({"metric": "smoothness_delta", "p95": delta.get("stabilized_step_p95"), "max": delta.get("stabilized_step_max"), "raw_p95": delta.get("raw_step_p95"), "raw_max": delta.get("raw_step_max")})
    return rows


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare raw and stabilized pseudo-3D predictions.")
    parser.add_argument("--raw-root", required=True, type=Path)
    parser.add_argument("--stabilized-root", required=True, type=Path)
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
