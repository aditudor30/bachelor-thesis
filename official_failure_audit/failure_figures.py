"""Diagnostic figures for Step 23A."""

from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Sequence

import numpy as np

from deep_oc_sort_3d.official_failure_audit.track1_parser import AuditTrack1Row


def write_failure_figures(
    predictions: Sequence[AuditTrack1Row], ground_truth: Sequence[AuditTrack1Row],
    original_details: Sequence[Dict[str, Any]], sweep: Dict[str, Any], output_root: Path,
) -> Dict[str, Any]:
    if not predictions or not ground_truth:
        return {"status": "insufficient_data", "files": []}
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return {"status": "matplotlib_not_available", "files": []}
    directory = output_root / "figures"
    directory.mkdir(parents=True, exist_ok=True)
    files = []

    names = ["x", "y", "z"]
    pred_values = [[getattr(row, name) for row in predictions] for name in names]
    gt_values = [[getattr(row, name) for row in ground_truth] for name in names]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for index, name in enumerate(names):
        axes[index].boxplot([pred_values[index], gt_values[index]], labels=["pred", "GT"], showfliers=False)
        axes[index].set_title(name.upper() + " range")
        axes[index].grid(alpha=0.2)
    files.append(_save(fig, directory / "pred_vs_gt_xyz_ranges.png", plt))

    ranked = list(sweep.get("individual", [])) + list(sweep.get("combined", []))
    ranked = sorted(ranked, key=lambda row: _number(row.get("match_rate_at_2m"), -1.0), reverse=True)[:20]
    labels = [str(row.get("hypothesis", "")) for row in ranked]
    centers = [_number(row.get("center_error_median"), np.nan) for row in ranked]
    rates = [_number(row.get("match_rate_at_2m"), 0.0) for row in ranked]
    fig, axis = plt.subplots(figsize=(12, 6))
    axis.barh(range(len(labels)), centers)
    axis.set_yticks(range(len(labels)))
    axis.set_yticklabels(labels, fontsize=8)
    axis.invert_yaxis()
    axis.set_xlabel("Median center error (m)")
    files.append(_save(fig, directory / "center_error_by_hypothesis.png", plt))
    fig, axis = plt.subplots(figsize=(12, 6))
    axis.barh(range(len(labels)), rates)
    axis.set_yticks(range(len(labels)))
    axis.set_yticklabels(labels, fontsize=8)
    axis.invert_yaxis()
    axis.set_xlabel("Match rate at 2 m")
    files.append(_save(fig, directory / "match_rate_by_hypothesis.png", plt))

    matched = [row for row in original_details if row.get("matched")]
    ratios = defaultdict(lambda: [[], [], []])
    yaw = defaultdict(list)
    for row in matched:
        class_id = str(row.get("class_id"))
        ratios[class_id][0].append(row.get("width_ratio_pred_over_gt"))
        ratios[class_id][1].append(row.get("length_ratio_pred_over_gt"))
        ratios[class_id][2].append(row.get("height_ratio_pred_over_gt"))
        yaw[class_id].append(row.get("yaw_error"))
    classes = sorted(ratios.keys(), key=int)
    positions = np.arange(len(classes), dtype=float)
    fig, axis = plt.subplots(figsize=(10, 5))
    for index, label in enumerate(["width", "length", "height"]):
        values = [_median(ratios[class_id][index]) for class_id in classes]
        axis.bar(positions + (index - 1) * 0.25, values, width=0.25, label=label)
    axis.set_xticks(positions)
    axis.set_xticklabels(classes)
    axis.set_xlabel("Official class id")
    axis.set_ylabel("Median pred / GT ratio")
    axis.legend()
    files.append(_save(fig, directory / "dimension_ratio_by_class.png", plt))
    fig, axis = plt.subplots(figsize=(10, 5))
    axis.bar(classes, [_median(yaw[class_id]) for class_id in classes])
    axis.set_xlabel("Official class id")
    axis.set_ylabel("Median yaw error (rad)")
    files.append(_save(fig, directory / "yaw_error_by_class.png", plt))
    return {"status": "ok", "files": files}


def _save(fig: Any, path: Path, plt: Any) -> str:
    fig.tight_layout()
    fig.savefig(str(path), dpi=160, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def _median(values: Sequence[Any]) -> float:
    array = np.asarray([float(value) for value in values if value is not None], dtype=float)
    array = array[np.isfinite(array)]
    return float(np.median(array)) if array.size else float("nan")


def _number(value: Any, default: float) -> float:
    try:
        return float(value) if value is not None else default
    except (TypeError, ValueError):
        return default
