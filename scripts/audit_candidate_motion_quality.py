"""Audit motion quality for MTMC candidates."""

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import yaml

from deep_oc_sort_3d.mtmc.candidate_io import read_candidates_file
from deep_oc_sort_3d.mtmc.candidate_motion_io import write_motion_metrics_csv
from deep_oc_sort_3d.mtmc.candidate_motion_quality import (
    CandidateMotionMetrics,
    compute_candidate_motion_metrics,
    merge_motion_quality_config,
)


def audit_candidate_motion_quality(args: Any) -> None:
    """Audit motion-quality metrics for a candidate file or folder."""
    config = _resolve_config(args)
    files = _resolve_candidate_files(args.candidates)
    all_metrics = []
    for path in _progress_iter(files, args.progress, "audit candidate motion files"):
        candidates = read_candidates_file(path)
        for candidate in _progress_iter(candidates, args.progress, "audit %s" % path.name):
            all_metrics.append(compute_candidate_motion_metrics(candidate, config))
    write_motion_metrics_csv(all_metrics, args.output_metrics)
    summary = summarize_motion_metrics(all_metrics)
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print_motion_summary(summary)
    print("Wrote metrics: %s" % args.output_metrics)
    print("Wrote summary: %s" % args.summary_output)


def summarize_motion_metrics(metrics_list: List[CandidateMotionMetrics]) -> Dict[str, Any]:
    """Build a motion-quality summary dictionary."""
    flags = _count_by(metrics_list, "motion_quality_flag")
    reasons = _count_by(metrics_list, "motion_reject_reason")
    max_steps = [item.max_step_distance_3d for item in metrics_list if item.max_step_distance_3d is not None]
    jumps = [item.jump_count for item in metrics_list]
    return {
        "total_candidates": len(metrics_list),
        "motion_good": flags.get("motion_good", 0),
        "motion_suspicious": flags.get("motion_suspicious", 0),
        "motion_invalid": flags.get("motion_invalid", 0),
        "motion_unknown": flags.get("motion_unknown", 0),
        "clean_count": len([item for item in metrics_list if item.is_motion_clean]),
        "motion_quality_flags": flags,
        "motion_reject_reasons": reasons,
        "per_class_flag_counts": _nested_count(metrics_list, "class_name", "motion_quality_flag"),
        "per_subset_flag_counts": _nested_count(metrics_list, "subset", "motion_quality_flag"),
        "max_step_stats": _stats(max_steps),
        "jump_count_stats": _stats(jumps),
        "top_worst_by_max_step": [_metrics_brief(item) for item in _top_by(metrics_list, "max_step_distance_3d", 20)],
        "top_worst_by_jump_count": [_metrics_brief(item) for item in _top_by(metrics_list, "jump_count", 20)],
    }


def print_motion_summary(summary: Dict[str, Any]) -> None:
    """Print compact motion-quality summary."""
    print("total candidates: %s" % summary.get("total_candidates"))
    print("motion_good: %s" % summary.get("motion_good"))
    print("motion_suspicious: %s" % summary.get("motion_suspicious"))
    print("motion_invalid: %s" % summary.get("motion_invalid"))
    print("motion_unknown: %s" % summary.get("motion_unknown"))
    print("clean count: %s" % summary.get("clean_count"))
    print("reject reasons: %s" % json.dumps(summary.get("motion_reject_reasons", {}), sort_keys=True))
    print("max step stats: %s" % json.dumps(summary.get("max_step_stats", {}), sort_keys=True))


def write_worst_outliers_csv(metrics_list: List[CandidateMotionMetrics], path: Path, top_k: int = 100) -> None:
    """Write worst motion outliers by max step and jump count."""
    rows = []
    selected = _top_by(metrics_list, "max_step_distance_3d", top_k)
    for item in selected:
        rows.append(_metrics_brief(item))
    fields = [
        "candidate_id",
        "subset",
        "scene_name",
        "camera_id",
        "class_name",
        "motion_quality_flag",
        "motion_reject_reason",
        "max_step_distance_3d",
        "max_speed_3d",
        "jump_count",
        "jump_ratio",
        "travel_distance_per_frame",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _resolve_config(args: Any) -> Dict[str, Any]:
    config = _load_config(args.config)
    if args.require_3d_motion is not None:
        config["require_3d_motion"] = bool(args.require_3d_motion)
    if args.allow_suspicious_as_clean:
        config["allow_suspicious_as_clean"] = True
    return merge_motion_quality_config(config)


def _load_config(path: Optional[Path]) -> Dict[str, Any]:
    if path is None:
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        return {}
    section = data.get("motion_quality", data)
    return section if isinstance(section, dict) else {}


def _resolve_candidate_files(path: Path) -> List[Path]:
    if path.is_file():
        return [path]
    jsonl_files = sorted(path.rglob("*_candidates.jsonl"))
    if jsonl_files:
        return jsonl_files
    return sorted(path.rglob("*_candidates.csv"))


def _count_by(metrics_list: List[CandidateMotionMetrics], field: str) -> Dict[str, int]:
    counts = {}
    for item in metrics_list:
        key = str(getattr(item, field))
        counts[key] = counts.get(key, 0) + 1
    return counts


def _nested_count(metrics_list: List[CandidateMotionMetrics], outer_field: str, inner_field: str) -> Dict[str, Dict[str, int]]:
    counts = {}
    for item in metrics_list:
        outer = str(getattr(item, outer_field))
        inner = str(getattr(item, inner_field))
        if outer not in counts:
            counts[outer] = {}
        counts[outer][inner] = counts[outer].get(inner, 0) + 1
    return counts


def _stats(values: List[Any]) -> Dict[str, Any]:
    if not values:
        return {"count": 0, "mean": None, "median": None, "p95": None, "max": None}
    arr = np.asarray(values, dtype=float)
    return {
        "count": int(arr.size),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "p95": float(np.percentile(arr, 95)),
        "max": float(np.max(arr)),
    }


def _top_by(metrics_list: List[CandidateMotionMetrics], field: str, top_k: int) -> List[CandidateMotionMetrics]:
    def key(item: CandidateMotionMetrics) -> float:
        value = getattr(item, field)
        if value is None:
            return -1.0
        return float(value)

    return sorted(metrics_list, key=key, reverse=True)[: int(top_k)]


def _metrics_brief(item: CandidateMotionMetrics) -> Dict[str, Any]:
    return {
        "candidate_id": item.candidate_id,
        "subset": item.subset,
        "scene_name": item.scene_name,
        "camera_id": item.camera_id,
        "class_name": item.class_name,
        "motion_quality_flag": item.motion_quality_flag,
        "motion_reject_reason": item.motion_reject_reason,
        "max_step_distance_3d": item.max_step_distance_3d,
        "max_speed_3d": item.max_speed_3d,
        "jump_count": item.jump_count,
        "jump_ratio": item.jump_ratio,
        "travel_distance_per_frame": item.travel_distance_per_frame,
    }


def _progress_iter(values: List[Any], show_progress: bool, desc: str) -> Iterable[Any]:
    if not show_progress:
        return values
    try:
        from tqdm import tqdm
    except ImportError:
        return _print_progress_iter(values, desc)
    return tqdm(values, desc=desc)


def _print_progress_iter(values: List[Any], desc: str) -> Iterable[Any]:
    total = len(values)
    for index, value in enumerate(values):
        if index == 0 or (index + 1) % 1000 == 0 or index + 1 == total:
            print("%s: %d/%d" % (desc, index + 1, total))
        yield value


def build_arg_parser() -> argparse.ArgumentParser:
    """Build CLI parser."""
    parser = argparse.ArgumentParser(description="Audit MTMC candidate motion quality.")
    parser.add_argument("--candidates", required=True, type=Path)
    parser.add_argument("--output-metrics", required=True, type=Path)
    parser.add_argument("--summary-output", required=True, type=Path)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--allow-suspicious-as-clean", action="store_true")
    req_group = parser.add_mutually_exclusive_group()
    req_group.add_argument("--require-3d-motion", dest="require_3d_motion", action="store_true", default=None)
    req_group.add_argument("--no-require-3d-motion", dest="require_3d_motion", action="store_false")
    progress_group = parser.add_mutually_exclusive_group()
    progress_group.add_argument("--progress", dest="progress", action="store_true")
    progress_group.add_argument("--no-progress", dest="progress", action="store_false")
    parser.set_defaults(progress=True)
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    audit_candidate_motion_quality(args)


if __name__ == "__main__":
    main()
