"""Compact comparison figures for Step 22C."""

from pathlib import Path
from typing import Any, Dict, List

from deep_oc_sort_3d.v4_geometry_refinement.geometry_refinement_config import output_root
from deep_oc_sort_3d.v4_geometry_refinement.track1_geometry_io import read_json


def write_geometry_figures(config: Dict[str, Any]) -> List[Path]:
    """Write the requested proxy and identity-comparison PNG figures."""
    import matplotlib.pyplot as plt

    root = output_root(config)
    comparison = read_json(root / "comparison" / "v4_geometry_refinement_summary.json")
    baseline = comparison.get("baseline_metrics", {})
    variants = comparison.get("variants", [])
    labels = ["V3.1"] + [str(row.get("variant", "unknown")).replace("v4_", "") for row in variants]
    figures_root = root / "figures"
    figures_root.mkdir(parents=True, exist_ok=True)
    specs = [
        ("step_p95_by_variant.png", "step_p95", "Step distance p95", "meters/frame"),
        ("suspect_tracks_by_variant.png", "suspect_track_count", "Suspect tracks", "count"),
        ("dimension_variance_by_variant.png", "dimension_variance_mean", "Mean dimension variance", "variance"),
        ("yaw_jump_count_by_variant.png", "yaw_jump_count", "Yaw jump count", "count"),
    ]
    paths: List[Path] = []
    for filename, metric, title, ylabel in specs:
        values = [_float_or_zero(baseline.get(metric))] + [_float_or_zero(row.get(metric)) for row in variants]
        path = figures_root / filename
        _bar_figure(plt, path, labels, values, title, ylabel)
        paths.append(path)
    rows = [_float_or_zero(baseline.get("rows"))] + [_float_or_zero(row.get("rows")) for row in variants]
    tracks = [_float_or_zero(baseline.get("unique_tracks"))] + [_float_or_zero(row.get("unique_tracks")) for row in variants]
    path = figures_root / "rows_tracks_comparison.png"
    figure, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].bar(range(len(labels)), rows, color="#2878B5")
    axes[0].set_title("Rows")
    axes[1].bar(range(len(labels)), tracks, color="#D95319")
    axes[1].set_title("Unique tracks")
    for axis in axes:
        axis.set_xticks(range(len(labels)))
        axis.set_xticklabels(labels, rotation=30, ha="right")
        axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(str(path), dpi=160)
    plt.close(figure)
    paths.append(path)
    return paths


def _bar_figure(plt: Any, path: Path, labels: List[str], values: List[float], title: str, ylabel: str) -> None:
    figure, axis = plt.subplots(figsize=(10, 5))
    axis.bar(range(len(labels)), values, color=["#555555"] + ["#2878B5"] * (len(labels) - 1))
    axis.set_xticks(range(len(labels)))
    axis.set_xticklabels(labels, rotation=30, ha="right")
    axis.set_title(title)
    axis.set_ylabel(ylabel)
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(str(path), dpi=160)
    plt.close(figure)


def _float_or_zero(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0
