"""Matplotlib plots for YOLO bbox audit CSV files."""

from pathlib import Path
from typing import Any, Dict, List

from deep_oc_sort_3d.detection2d.yolo_bbox_audit import load_bbox_audit_csv


def plot_area_distribution_by_class(audit_csv: Path, output_path: Path) -> None:
    """Plot area_norm distribution by class."""
    rows = load_bbox_audit_csv(audit_csv)
    class_values = _group_values(rows, "class_name", "area_norm")
    _boxplot(class_values, "area_norm by class", "area_norm", output_path)


def plot_difficulty_distribution_by_class(audit_csv: Path, output_path: Path) -> None:
    """Plot stacked difficulty counts by class."""
    rows = load_bbox_audit_csv(audit_csv)
    counts = {}
    for row in rows:
        class_name = str(row["class_name"])
        difficulty = str(row["difficulty"])
        if class_name not in counts:
            counts[class_name] = {"easy": 0, "medium": 0, "hard": 0}
        counts[class_name][difficulty] = counts[class_name].get(difficulty, 0) + 1
    _bar_group_counts(counts, "difficulty by class", output_path)


def plot_class_counts_by_scene(audit_csv: Path, output_path: Path) -> None:
    """Plot class counts grouped by scene."""
    rows = load_bbox_audit_csv(audit_csv)
    counts = _nested_counts(rows, "scene_name", "class_name")
    _bar_group_counts(counts, "class counts by scene", output_path)


def plot_class_counts_by_camera(audit_csv: Path, output_path: Path) -> None:
    """Plot class counts grouped by camera."""
    rows = load_bbox_audit_csv(audit_csv)
    counts = _nested_counts(rows, "camera_id", "class_name")
    _bar_group_counts(counts, "class counts by camera", output_path)


def _boxplot(values_by_name: Dict[str, List[float]], title: str, ylabel: str, output_path: Path) -> None:
    import matplotlib.pyplot as plt

    names = sorted(values_by_name.keys())
    values = [values_by_name[name] for name in names]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(max(8, len(names) * 1.2), 5))
    plt.boxplot(values, labels=names, showfliers=False)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(str(output_path))
    plt.close()


def _bar_group_counts(counts: Dict[str, Dict[str, int]], title: str, output_path: Path) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    groups = sorted(counts.keys())
    labels = sorted(set(label for group in counts.values() for label in group.keys()))
    x = np.arange(len(groups))
    bottom = np.zeros(len(groups))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(max(8, len(groups) * 0.8), 5))
    for label in labels:
        values = np.asarray([counts[group].get(label, 0) for group in groups], dtype=float)
        plt.bar(x, values, bottom=bottom, label=label)
        bottom += values
    plt.title(title)
    plt.ylabel("count")
    plt.xticks(x, groups, rotation=45, ha="right")
    plt.legend()
    plt.tight_layout()
    plt.savefig(str(output_path))
    plt.close()


def _group_values(rows: List[Dict[str, Any]], group_key: str, value_key: str) -> Dict[str, List[float]]:
    grouped = {}
    for row in rows:
        group = str(row[group_key])
        if group not in grouped:
            grouped[group] = []
        grouped[group].append(float(row[value_key]))
    return grouped


def _nested_counts(rows: List[Dict[str, Any]], group_key: str, label_key: str) -> Dict[str, Dict[str, int]]:
    counts = {}
    for row in rows:
        group = str(row[group_key])
        label = str(row[label_key])
        if group not in counts:
            counts[group] = {}
        counts[group][label] = counts[group].get(label, 0) + 1
    return counts

