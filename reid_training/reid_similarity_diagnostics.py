"""Similarity diagnostics and fine-tuning verdicts for ReID embeddings."""

import random
from typing import Any, Dict, List, Optional

import numpy as np


def compute_similarity_diagnostics(
    embeddings: np.ndarray,
    metadata: List[Dict[str, Any]],
    max_pairs: int = 200000,
    thresholds: Optional[List[float]] = None,
    seed: int = 42,
) -> Dict[str, Any]:
    """Compute same/different identity similarity diagnostics."""
    if thresholds is None:
        thresholds = [0.75, 0.80, 0.85]
    if embeddings.size == 0 or len(metadata) < 2:
        return {"status": "empty", "num_pairs": 0}
    matrix = _l2_normalize_matrix(embeddings)
    labels = [str(row.get("identity_id", "")) for row in metadata]
    pairs = _sample_pairs(len(labels), int(max_pairs), seed)
    same_values: List[float] = []
    diff_values: List[float] = []
    for left, right in pairs:
        value = float(np.dot(matrix[left], matrix[right]))
        if labels[left] == labels[right]:
            same_values.append(value)
        else:
            diff_values.append(value)
    summary: Dict[str, Any] = {
        "status": "ok",
        "num_pairs": len(pairs),
        "same_count": len(same_values),
        "different_count": len(diff_values),
        "same_gt": _stats(same_values),
        "different_gt": _stats(diff_values),
    }
    same_mean = summary["same_gt"].get("mean")
    diff_mean = summary["different_gt"].get("mean")
    summary["separation_margin"] = None if same_mean is None or diff_mean is None else float(same_mean) - float(diff_mean)
    for threshold in thresholds:
        key = "threshold_%.2f" % float(threshold)
        summary[key] = {
            "same_recall": _fraction_at_least(same_values, float(threshold)),
            "different_high_sim_risk": _fraction_at_least(diff_values, float(threshold)),
        }
    return summary


def metric_deltas(pretrained: Dict[str, Any], finetuned: Dict[str, Any], pretrained_sim: Dict[str, Any], finetuned_sim: Dict[str, Any]) -> Dict[str, Any]:
    """Compute metric deltas from pretrained to fine-tuned."""
    return {
        "top1_delta": _delta(pretrained, finetuned, "top1_accuracy"),
        "top5_delta": _delta(pretrained, finetuned, "top5_accuracy"),
        "top10_delta": _delta(pretrained, finetuned, "top10_accuracy"),
        "mAP_delta": _delta(pretrained, finetuned, "mAP"),
        "separation_margin_delta": _value(finetuned_sim.get("separation_margin")) - _value(pretrained_sim.get("separation_margin")),
        "different_risk_080_delta": _threshold_delta(pretrained_sim, finetuned_sim, "threshold_0.80", "different_high_sim_risk"),
        "same_recall_080_delta": _threshold_delta(pretrained_sim, finetuned_sim, "threshold_0.80", "same_recall"),
    }


def finetuning_verdict(pretrained: Dict[str, Any], finetuned: Dict[str, Any], deltas: Dict[str, Any]) -> Dict[str, Any]:
    """Classify fine-tuned ReID readiness."""
    top1_delta = _value(deltas.get("top1_delta"))
    top5_delta = _value(deltas.get("top5_delta"))
    margin_delta = _value(deltas.get("separation_margin_delta"))
    risk_delta = _value(deltas.get("different_risk_080_delta"))
    if top1_delta >= 0.03 and top5_delta >= 0.01 and margin_delta >= 0.03 and risk_delta <= 0.0:
        verdict = "finetuned_reid_ready_for_association"
    elif top1_delta >= 0.01 or margin_delta >= 0.02:
        verdict = "finetuned_reid_promising_needs_threshold_tuning"
    elif top1_delta < -0.02 or margin_delta < -0.02:
        verdict = "finetuned_reid_overfit_or_invalid"
    else:
        verdict = "finetuned_reid_no_clear_gain"
    return {
        "verdict": verdict,
        "top1_delta": top1_delta,
        "top5_delta": top5_delta,
        "separation_margin_delta": margin_delta,
        "different_risk_080_delta": risk_delta,
        "pretrained_top1": pretrained.get("top1_accuracy"),
        "finetuned_top1": finetuned.get("top1_accuracy"),
    }


def _sample_pairs(length: int, max_pairs: int, seed: int) -> List[Any]:
    rng = random.Random(seed)
    total_possible = int(length) * int(length - 1) // 2
    if total_possible <= max_pairs:
        pairs = []
        for left in range(length):
            for right in range(left + 1, length):
                pairs.append((left, right))
        return pairs
    pairs_set = set()
    while len(pairs_set) < int(max_pairs):
        left = rng.randrange(length)
        right = rng.randrange(length)
        if left == right:
            continue
        if left > right:
            left, right = right, left
        pairs_set.add((left, right))
    return list(pairs_set)


def _stats(values: List[float]) -> Dict[str, Any]:
    if not values:
        return {"count": 0, "mean": None, "median": None, "p05": None, "p95": None}
    arr = np.asarray(values, dtype=float)
    return {
        "count": int(arr.size),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "p05": float(np.percentile(arr, 5)),
        "p95": float(np.percentile(arr, 95)),
    }


def _fraction_at_least(values: List[float], threshold: float) -> Any:
    if not values:
        return None
    arr = np.asarray(values, dtype=float)
    return float(np.mean(arr >= float(threshold)))


def _l2_normalize_matrix(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms <= 1e-12] = 1.0
    return matrix / norms


def _delta(left: Dict[str, Any], right: Dict[str, Any], key: str) -> Any:
    if left.get(key) is None or right.get(key) is None:
        return None
    return float(right.get(key)) - float(left.get(key))


def _threshold_delta(left: Dict[str, Any], right: Dict[str, Any], threshold_key: str, metric_key: str) -> Any:
    left_value = (left.get(threshold_key) or {}).get(metric_key)
    right_value = (right.get(threshold_key) or {}).get(metric_key)
    if left_value is None or right_value is None:
        return None
    return float(right_value) - float(left_value)


def _value(value: Any) -> float:
    if value is None:
        return 0.0
    return float(value)
