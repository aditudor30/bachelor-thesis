"""Diagnostic evaluation for MTMC candidate sets."""

from typing import Any, Dict, List

import numpy as np

from deep_oc_sort_3d.mtmc.candidate_types import MTMCTrackletCandidate


def evaluate_candidate_set(candidates: List[MTMCTrackletCandidate]) -> Dict[str, Any]:
    """Evaluate candidate quality using diagnostics only."""
    kept = [item for item in candidates if item.is_candidate]
    with_gt = [item for item in kept if item.majority_gt_object_id is not None]
    purity_values = [float(item.gt_purity) for item in with_gt if item.gt_purity is not None]
    metrics = {
        "num_candidates": len(candidates),
        "kept_candidates": len(kept),
        "candidates_with_gt": len(with_gt),
        "purity_mean": _mean(purity_values),
        "purity_median": _median(purity_values),
        "per_class_purity": _per_class_purity(kept),
        "per_class_kept": _count_by(kept, "class_name"),
        "gt_object_coverage_diagnostic": len(set(int(item.majority_gt_object_id) for item in with_gt)),
        "rejected_with_good_purity_count": len(
            [
                item
                for item in candidates
                if not item.is_candidate and item.gt_purity is not None and float(item.gt_purity) >= 0.9
            ]
        ),
    }
    if not with_gt:
        metrics["gt_note"] = "No GT ids available; GT diagnostics are None/0 for this input."
    return metrics


def _per_class_purity(candidates: List[MTMCTrackletCandidate]) -> Dict[str, Any]:
    values = {}
    for candidate in candidates:
        if candidate.gt_purity is None:
            continue
        key = str(candidate.class_name)
        if key not in values:
            values[key] = []
        values[key].append(float(candidate.gt_purity))
    return {key: _mean(items) for key, items in values.items()}


def _count_by(candidates: List[MTMCTrackletCandidate], field: str) -> Dict[str, int]:
    counts = {}
    for candidate in candidates:
        key = str(getattr(candidate, field))
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
