"""Summary utilities for MTMC candidates."""

import csv
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from deep_oc_sort_3d.mtmc.candidate_types import MTMCTrackletCandidate


def summarize_candidates(candidates: List[MTMCTrackletCandidate]) -> Dict[str, Any]:
    """Summarize kept and rejected MTMC candidates."""
    kept = [item for item in candidates if item.is_candidate]
    rejected = [item for item in candidates if not item.is_candidate]
    purity_values = [float(item.gt_purity) for item in candidates if item.gt_purity is not None]
    return {
        "total_candidates_including_rejected": len(candidates),
        "kept_candidates": len(kept),
        "rejected_candidates": len(rejected),
        "kept_ratio": _ratio(len(kept), len(candidates)),
        "per_subset_counts": _count_by(candidates, "subset"),
        "per_class_counts": _count_by(candidates, "class_name"),
        "per_class_kept_counts": _count_by(kept, "class_name"),
        "per_scene_counts": _count_by(candidates, "scene_name"),
        "per_camera_counts": _count_by(candidates, "camera_id"),
        "reject_reason_counts": _reject_reason_counts(candidates),
        "quality_flag_counts": _count_by(candidates, "quality_flag"),
        "has_3d_count": len([item for item in candidates if item.has_3d]),
        "no_3d_count": len([item for item in candidates if not item.has_3d]),
        "mean_length": _mean([item.length for item in candidates]),
        "median_length": _median([item.length for item in candidates]),
        "mean_confidence": _mean([item.mean_confidence for item in candidates]),
        "candidate_mean_confidence": _mean([item.mean_confidence for item in kept]),
        "candidate_mean_length": _mean([item.length for item in kept]),
        "diagnostic_purity_mean": _mean(purity_values),
    }


def write_candidate_summary_json(summary: Dict[str, Any], path: Path) -> None:
    """Write candidate summary as JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")


def write_candidate_summary_csv(summary: Dict[str, Any], path: Path) -> None:
    """Write candidate summary as compact metric CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["metric", "value"])
        writer.writeheader()
        for key, value in summary.items():
            if isinstance(value, (dict, list)):
                value = json.dumps(value, sort_keys=True)
            writer.writerow({"metric": key, "value": value})


def print_candidate_summary(summary: Dict[str, Any]) -> None:
    """Print a compact candidate summary."""
    print("total candidates including rejected: %s" % summary.get("total_candidates_including_rejected"))
    print("kept candidates: %s" % summary.get("kept_candidates"))
    print("rejected candidates: %s" % summary.get("rejected_candidates"))
    print("kept ratio: %s" % summary.get("kept_ratio"))
    print("has 3d: %s" % summary.get("has_3d_count"))
    print("no 3d: %s" % summary.get("no_3d_count"))
    print("mean length: %s" % summary.get("mean_length"))
    print("candidate mean length: %s" % summary.get("candidate_mean_length"))
    print("diagnostic purity mean: %s" % summary.get("diagnostic_purity_mean"))
    print("per class kept: %s" % json.dumps(summary.get("per_class_kept_counts", {}), sort_keys=True))
    print("reject reasons: %s" % json.dumps(summary.get("reject_reason_counts", {}), sort_keys=True))


def _count_by(candidates: List[MTMCTrackletCandidate], field: str) -> Dict[str, int]:
    counts = {}
    for candidate in candidates:
        key = str(getattr(candidate, field))
        counts[key] = counts.get(key, 0) + 1
    return counts


def _reject_reason_counts(candidates: List[MTMCTrackletCandidate]) -> Dict[str, int]:
    counts = {}
    for candidate in candidates:
        key = "ok" if candidate.reject_reason is None else str(candidate.reject_reason)
        counts[key] = counts.get(key, 0) + 1
    return counts


def _mean(values: List[Any]) -> Any:
    if not values:
        return None
    return float(np.mean(np.asarray(values, dtype=float)))


def _median(values: List[Any]) -> Any:
    if not values:
        return None
    return float(np.median(np.asarray(values, dtype=float)))


def _ratio(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return float(numerator) / float(denominator)
