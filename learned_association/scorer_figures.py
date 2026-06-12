"""Optional matplotlib figures for Step 20B scorer comparison."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from deep_oc_sort_3d.learned_association.scorer_evaluator import (
    average_precision_binary,
    calibration_metrics,
    roc_auc_score_binary,
)


def generate_scorer_figures(
    score_map: Dict[str, np.ndarray],
    labels: np.ndarray,
    output_dir: Path,
    threshold_rows: Sequence[Dict[str, Any]],
    history: Optional[Sequence[Dict[str, Any]]] = None,
    feature_importance: Optional[Dict[str, Dict[str, float]]] = None,
    per_camera_rows: Optional[Sequence[Dict[str, Any]]] = None,
    enabled: bool = True,
) -> List[str]:
    """Generate diagnostic figures without making matplotlib mandatory."""
    if not enabled:
        return ["figures_disabled"]
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return ["matplotlib_unavailable_figures_skipped"]
    output_dir.mkdir(parents=True, exist_ok=True)
    warnings = []  # type: List[str]
    try:
        _ranking_curves(plt, score_map, labels, output_dir)
        _threshold_figure(plt, threshold_rows, output_dir / "threshold_precision_recall.png")
        _score_distribution(plt, score_map, labels, output_dir / "score_distribution_pos_neg.png")
        _calibration_figure(plt, score_map, labels, output_dir / "calibration_curve.png")
        _training_loss(plt, history or [], output_dir / "mlp_train_val_loss.png")
        _feature_importance(plt, feature_importance or {}, output_dir / "feature_importance.png")
        _per_camera_auc(plt, per_camera_rows or [], output_dir / "per_camera_pair_auc.png")
    except Exception as exc:
        warnings.append("scorer_figure_generation_failed: %s" % exc)
    return warnings


def _ranking_curves(plt: Any, score_map: Dict[str, np.ndarray], labels: np.ndarray, output_dir: Path) -> None:
    figure_roc, axis_roc = plt.subplots(figsize=(7, 6))
    figure_pr, axis_pr = plt.subplots(figsize=(7, 6))
    for name, scores in score_map.items():
        fpr, tpr = _roc_points(labels, scores)
        recall, precision = _pr_points(labels, scores)
        auc = roc_auc_score_binary(labels, scores)
        ap = average_precision_binary(labels, scores)
        axis_roc.plot(fpr, tpr, label="%s AUC=%.3f" % (name, auc or 0.0))
        axis_pr.plot(recall, precision, label="%s AP=%.3f" % (name, ap or 0.0))
    axis_roc.plot([0, 1], [0, 1], linestyle="--", color="gray")
    axis_roc.set_xlabel("false positive rate")
    axis_roc.set_ylabel("true positive rate")
    axis_roc.set_title("ROC curves")
    axis_roc.legend(fontsize=8)
    axis_pr.set_xlabel("recall")
    axis_pr.set_ylabel("precision")
    axis_pr.set_title("Precision-recall curves")
    axis_pr.legend(fontsize=8)
    figure_roc.tight_layout()
    figure_pr.tight_layout()
    figure_roc.savefig(str(output_dir / "roc_curves.png"), dpi=160)
    figure_pr.savefig(str(output_dir / "precision_recall_curves.png"), dpi=160)
    plt.close(figure_roc)
    plt.close(figure_pr)


def _threshold_figure(plt: Any, rows: Sequence[Dict[str, Any]], path: Path) -> None:
    figure, axis = plt.subplots(figsize=(9, 6))
    names = sorted({str(row.get("model_name")) for row in rows})
    for name in names:
        selected = sorted(
            [row for row in rows if str(row.get("model_name")) == name],
            key=lambda row: float(row.get("threshold", 0.0)),
        )
        thresholds = [float(row.get("threshold", 0.0)) for row in selected]
        precision = [float(row.get("precision") or 0.0) for row in selected]
        recall = [float(row.get("recall") or 0.0) for row in selected]
        axis.plot(thresholds, precision, label=name + " precision")
        axis.plot(thresholds, recall, linestyle="--", label=name + " recall")
    axis.set_xlabel("threshold")
    axis.set_ylabel("metric")
    axis.set_ylim(0.0, 1.02)
    axis.set_title("Threshold precision and recall")
    axis.legend(fontsize=7, ncol=2)
    figure.tight_layout()
    figure.savefig(str(path), dpi=160)
    plt.close(figure)


def _score_distribution(plt: Any, score_map: Dict[str, np.ndarray], labels: np.ndarray, path: Path) -> None:
    names = list(score_map.keys())
    figure, axes = plt.subplots(max(1, len(names)), 1, figsize=(8, 3 * max(1, len(names))), squeeze=False)
    for index, name in enumerate(names):
        axis = axes[index][0]
        scores = score_map[name]
        axis.hist(scores[labels == 1], bins=40, alpha=0.55, density=True, label="positive")
        axis.hist(scores[labels == 0], bins=40, alpha=0.55, density=True, label="negative")
        axis.set_title(name)
        axis.legend()
    figure.tight_layout()
    figure.savefig(str(path), dpi=160)
    plt.close(figure)


def _calibration_figure(plt: Any, score_map: Dict[str, np.ndarray], labels: np.ndarray, path: Path) -> None:
    figure, axis = plt.subplots(figsize=(7, 6))
    for name, scores in score_map.items():
        metrics = calibration_metrics(labels, scores)
        valid = [row for row in metrics["bins"] if row["count"] > 0]
        axis.plot(
            [row["mean_score"] for row in valid],
            [row["positive_rate"] for row in valid],
            marker="o",
            label=name,
        )
    axis.plot([0, 1], [0, 1], linestyle="--", color="gray")
    axis.set_xlabel("mean predicted score")
    axis.set_ylabel("positive rate")
    axis.set_title("Calibration curve")
    axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(str(path), dpi=160)
    plt.close(figure)


def _training_loss(plt: Any, history: Sequence[Dict[str, Any]], path: Path) -> None:
    figure, axis = plt.subplots(figsize=(8, 5))
    if history:
        axis.plot([row["epoch"] for row in history], [row["train_loss"] for row in history], label="train")
        axis.plot([row["epoch"] for row in history], [row["val_loss"] for row in history], label="val")
        axis.legend()
    else:
        axis.text(0.5, 0.5, "No MLP training history", ha="center", va="center")
    axis.set_xlabel("epoch")
    axis.set_ylabel("loss")
    axis.set_title("MLP train/validation loss")
    figure.tight_layout()
    figure.savefig(str(path), dpi=160)
    plt.close(figure)


def _feature_importance(plt: Any, importance: Dict[str, Dict[str, float]], path: Path) -> None:
    selected_name = ""
    selected_values = {}  # type: Dict[str, float]
    for name, values in importance.items():
        if len(values) > len(selected_values):
            selected_name = name
            selected_values = values
    figure, axis = plt.subplots(figsize=(9, 7))
    if selected_values:
        top = sorted(selected_values.items(), key=lambda item: item[1], reverse=True)[:20]
        labels = [item[0] for item in reversed(top)]
        values = [item[1] for item in reversed(top)]
        axis.barh(range(len(labels)), values)
        axis.set_yticks(range(len(labels)))
        axis.set_yticklabels(labels, fontsize=8)
        axis.set_title("Feature importance: %s" % selected_name)
    else:
        axis.text(0.5, 0.5, "Feature importance unavailable", ha="center", va="center")
    figure.tight_layout()
    figure.savefig(str(path), dpi=160)
    plt.close(figure)


def _per_camera_auc(plt: Any, rows: Sequence[Dict[str, Any]], path: Path) -> None:
    figure, axis = plt.subplots(figsize=(11, 7))
    valid = [row for row in rows if row.get("roc_auc") not in (None, "", "not_available")]
    valid.sort(key=lambda row: float(row.get("roc_auc", 0.0)), reverse=True)
    valid = valid[:30]
    if valid:
        labels = ["%s:%s" % (row.get("model_name"), row.get("camera_pair")) for row in reversed(valid)]
        values = [float(row.get("roc_auc", 0.0)) for row in reversed(valid)]
        axis.barh(range(len(labels)), values)
        axis.set_yticks(range(len(labels)))
        axis.set_yticklabels(labels, fontsize=7)
        axis.set_xlim(0.0, 1.0)
    else:
        axis.text(0.5, 0.5, "Per-camera-pair AUC unavailable", ha="center", va="center")
    axis.set_title("Top per-camera-pair ROC-AUC")
    figure.tight_layout()
    figure.savefig(str(path), dpi=160)
    plt.close(figure)


def _roc_points(labels: np.ndarray, scores: np.ndarray) -> Any:
    thresholds = np.r_[np.inf, np.sort(np.unique(scores))[::-1], -np.inf]
    points = []
    positives = max(1, int(np.sum(labels == 1)))
    negatives = max(1, int(np.sum(labels == 0)))
    for threshold in thresholds:
        predictions = scores >= threshold
        tp = int(np.sum(predictions & (labels == 1)))
        fp = int(np.sum(predictions & (labels == 0)))
        points.append((fp / float(negatives), tp / float(positives)))
    return np.asarray([item[0] for item in points]), np.asarray([item[1] for item in points])


def _pr_points(labels: np.ndarray, scores: np.ndarray) -> Any:
    order = np.argsort(-scores, kind="mergesort")
    sorted_labels = labels[order]
    tp = np.cumsum(sorted_labels == 1)
    fp = np.cumsum(sorted_labels == 0)
    precision = tp / np.maximum(1, tp + fp)
    recall = tp / float(max(1, int(np.sum(labels == 1))))
    return recall, precision
