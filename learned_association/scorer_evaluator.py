"""Dependency-light binary metrics and grouped scorer evaluation."""

from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


def sigmoid_scores(logits: Any) -> np.ndarray:
    """Convert logits to stable probabilities."""
    values = np.asarray(logits, dtype=np.float64)
    return 1.0 / (1.0 + np.exp(-np.clip(values, -50.0, 50.0)))


def binary_classification_metrics(
    labels: Any, scores: Any, threshold: float = 0.5
) -> Dict[str, Any]:
    """Compute thresholded and ranking metrics for binary labels."""
    y_true, y_score = _validated_arrays(labels, scores)
    predictions = (y_score >= float(threshold)).astype(np.int64)
    tp = int(np.sum((predictions == 1) & (y_true == 1)))
    fp = int(np.sum((predictions == 1) & (y_true == 0)))
    tn = int(np.sum((predictions == 0) & (y_true == 0)))
    fn = int(np.sum((predictions == 0) & (y_true == 1)))
    precision = _ratio(tp, tp + fp)
    recall = _ratio(tp, tp + fn)
    specificity = _ratio(tn, tn + fp)
    fpr = _ratio(fp, fp + tn)
    fnr = _ratio(fn, fn + tp)
    accuracy = _ratio(tp + tn, len(y_true))
    f1 = None
    if precision is not None and recall is not None and precision + recall > 0.0:
        f1 = 2.0 * precision * recall / (precision + recall)
    balanced_accuracy = None
    if recall is not None and specificity is not None:
        balanced_accuracy = (recall + specificity) / 2.0
    return {
        "num_pairs": int(len(y_true)),
        "positives": int(np.sum(y_true == 1)),
        "negatives": int(np.sum(y_true == 0)),
        "threshold": float(threshold),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "balanced_accuracy": balanced_accuracy,
        "false_positive_rate": fpr,
        "false_negative_rate": fnr,
        "roc_auc": roc_auc_score_binary(y_true, y_score),
        "pr_auc": average_precision_binary(y_true, y_score),
    }


def threshold_sweep_metrics(
    model_name: str,
    labels: Any,
    scores: Any,
    thresholds: Sequence[float],
) -> List[Dict[str, Any]]:
    """Evaluate one model over fixed thresholds."""
    rows = []
    for threshold in thresholds:
        row = binary_classification_metrics(labels, scores, float(threshold))
        row["model_name"] = model_name
        rows.append(row)
    return rows


def grouped_metrics(
    labels: Any,
    scores: Any,
    metadata: Sequence[Dict[str, Any]],
    group_key: str,
    threshold: float,
    model_name: str,
) -> List[Dict[str, Any]]:
    """Compute metrics per scene or camera pair."""
    y_true, y_score = _validated_arrays(labels, scores)
    groups = defaultdict(list)  # type: Dict[str, List[int]]
    for index, row in enumerate(metadata):
        groups[str(row.get(group_key) or "unknown")].append(index)
    output = []
    for group_name in sorted(groups.keys()):
        indices = np.asarray(groups[group_name], dtype=np.int64)
        metrics = binary_classification_metrics(y_true[indices], y_score[indices], threshold)
        metrics["model_name"] = model_name
        metrics[group_key] = group_name
        output.append(metrics)
    return output


def hard_negative_metrics(
    labels: Any,
    scores: Any,
    metadata: Sequence[Dict[str, Any]],
    threshold: float,
) -> Dict[str, Any]:
    """Measure false positives among rows marked as hard negatives."""
    y_true, y_score = _validated_arrays(labels, scores)
    indices = [
        index
        for index, row in enumerate(metadata)
        if str(row.get("hard_negative", "0")) in ("1", "True", "true") and y_true[index] == 0
    ]
    if not indices:
        return {
            "hard_negative_count": 0,
            "hard_negative_false_positives": 0,
            "hard_negative_false_positive_rate": None,
        }
    selected = y_score[np.asarray(indices, dtype=np.int64)]
    false_positives = int(np.sum(selected >= float(threshold)))
    return {
        "hard_negative_count": len(indices),
        "hard_negative_false_positives": false_positives,
        "hard_negative_false_positive_rate": false_positives / float(len(indices)),
    }


def calibration_metrics(labels: Any, scores: Any, bins: int = 10) -> Dict[str, Any]:
    """Compute Brier score and equal-width expected calibration error."""
    y_true, y_score = _validated_arrays(labels, scores)
    brier = float(np.mean((y_score - y_true) ** 2))
    rows = []
    ece = 0.0
    boundaries = np.linspace(0.0, 1.0, int(bins) + 1)
    for index in range(int(bins)):
        low = boundaries[index]
        high = boundaries[index + 1]
        mask = (y_score >= low) & (y_score <= high if index == bins - 1 else y_score < high)
        count = int(np.sum(mask))
        if count == 0:
            rows.append({"bin_start": low, "bin_end": high, "count": 0, "mean_score": None, "positive_rate": None})
            continue
        mean_score = float(np.mean(y_score[mask]))
        positive_rate = float(np.mean(y_true[mask]))
        ece += count / float(len(y_true)) * abs(mean_score - positive_rate)
        rows.append(
            {
                "bin_start": float(low),
                "bin_end": float(high),
                "count": count,
                "mean_score": mean_score,
                "positive_rate": positive_rate,
            }
        )
    return {"brier_score": brier, "expected_calibration_error": float(ece), "bins": rows}


def top_error_rows(
    labels: Any,
    scores: Any,
    metadata: Sequence[Dict[str, Any]],
    limit: int = 200,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Return highest-scoring negatives and lowest-scoring positives."""
    y_true, y_score = _validated_arrays(labels, scores)
    false_positive_candidates = []
    false_negative_candidates = []
    for index, row in enumerate(metadata):
        payload = dict(row)
        payload["score"] = float(y_score[index])
        payload["spatial_distance"] = row.get("center_mean_distance_3d")
        payload["reason"] = "high_score_negative" if y_true[index] == 0 else "low_score_positive"
        if y_true[index] == 0:
            false_positive_candidates.append(payload)
        else:
            false_negative_candidates.append(payload)
    false_positive_candidates.sort(key=lambda row: float(row["score"]), reverse=True)
    false_negative_candidates.sort(key=lambda row: float(row["score"]))
    return false_positive_candidates[:limit], false_negative_candidates[:limit]


def roc_auc_score_binary(labels: Any, scores: Any) -> Optional[float]:
    """Compute ROC-AUC with average ranks for ties."""
    y_true, y_score = _validated_arrays(labels, scores)
    positives = int(np.sum(y_true == 1))
    negatives = int(np.sum(y_true == 0))
    if positives == 0 or negatives == 0:
        return None
    ranks = _average_ranks(y_score)
    positive_rank_sum = float(np.sum(ranks[y_true == 1]))
    auc = (positive_rank_sum - positives * (positives + 1) / 2.0) / float(positives * negatives)
    return float(auc)


def average_precision_binary(labels: Any, scores: Any) -> Optional[float]:
    """Compute non-interpolated average precision."""
    y_true, y_score = _validated_arrays(labels, scores)
    positives = int(np.sum(y_true == 1))
    if positives == 0:
        return None
    order = np.argsort(-y_score, kind="mergesort")
    sorted_labels = y_true[order]
    cumulative = np.cumsum(sorted_labels == 1)
    positive_positions = np.where(sorted_labels == 1)[0]
    precisions = cumulative[positive_positions] / (positive_positions + 1.0)
    return float(np.mean(precisions))


def _validated_arrays(labels: Any, scores: Any) -> Tuple[np.ndarray, np.ndarray]:
    y_true = np.asarray(labels, dtype=np.int64).reshape(-1)
    y_score = np.asarray(scores, dtype=np.float64).reshape(-1)
    if y_true.shape != y_score.shape:
        raise ValueError("labels and scores must have identical shapes")
    if not np.all(np.isfinite(y_score)):
        raise ValueError("scores contain NaN or infinity")
    return y_true, y_score


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        average_rank = (start + 1 + end) / 2.0
        ranks[order[start:end]] = average_rank
        start = end
    return ranks


def _ratio(numerator: int, denominator: int) -> Optional[float]:
    return float(numerator) / float(denominator) if denominator > 0 else None
