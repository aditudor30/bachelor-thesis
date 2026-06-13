"""Diagnostic before/after and identity figures for Step 22D."""

from pathlib import Path
from typing import Any, Dict, List

from deep_oc_sort_3d.v5_geometry_calibration.geometry_calibration_config import output_root
from deep_oc_sort_3d.v5_geometry_calibration.geometry_calibration_io import read_csv, read_json


def write_v5_figures(config: Dict[str, Any]) -> List[Path]:
    """Write requested calibration figures from official_val diagnostics."""
    import matplotlib.pyplot as plt

    root = output_root(config)
    rows = read_csv(root / "validation_diagnostics" / "official_val_eval_before_after.csv")
    figures = root / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    specs = [
        ("center_error_before_after.png", "center_error_mean", "Center error mean"),
        ("dimension_error_before_after.png", "dimension_error_mean", "Dimension error mean"),
        ("yaw_error_before_after.png", "yaw_error_mean", "Yaw error mean"),
    ]
    output: List[Path] = []
    for filename, metric, title in specs:
        selected = [row for row in rows if row.get("metric") == metric]
        path = figures / filename
        _before_after_plot(plt, path, selected, title)
        output.append(path)
    per_class = read_csv(root / "validation_diagnostics" / "per_class_before_after.csv")
    path = figures / "per_class_improvement.png"
    _per_class_plot(plt, path, per_class)
    output.append(path)
    comparison = read_json(root / "comparison" / "v5_geometry_calibration_summary.json")
    path = figures / "rows_tracks_comparison.png"
    _rows_tracks_plot(plt, path, comparison.get("variants", []))
    output.append(path)
    return output


def _before_after_plot(plt: Any, path: Path, rows: List[Dict[str, Any]], title: str) -> None:
    labels = [str(row.get("variant", "")) for row in rows]
    before = [_number(row.get("before")) for row in rows]
    after = [_number(row.get("after")) for row in rows]
    figure, axis = plt.subplots(figsize=(11, 5))
    x = list(range(len(labels)))
    axis.bar([value - 0.2 for value in x], before, width=0.4, label="before")
    axis.bar([value + 0.2 for value in x], after, width=0.4, label="after")
    axis.set_xticks(x)
    axis.set_xticklabels(labels, rotation=30, ha="right")
    axis.set_title(title)
    axis.legend()
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(str(path), dpi=160)
    plt.close(figure)


def _per_class_plot(plt: Any, path: Path, rows: List[Dict[str, Any]]) -> None:
    selected = [row for row in rows if row.get("metric") in ("center_error_mean", "dimension_error_mean", "yaw_error_mean")]
    labels = ["%s:%s" % (row.get("official_class_id"), str(row.get("metric", "")).replace("_error_mean", "")) for row in selected]
    deltas = [_number(row.get("delta")) for row in selected]
    figure, axis = plt.subplots(figsize=(14, 6))
    axis.bar(range(len(labels)), deltas, color=["#2878B5" if value <= 0.0 else "#D95319" for value in deltas])
    axis.axhline(0.0, color="black", linewidth=1)
    axis.set_xticks(range(len(labels)))
    axis.set_xticklabels(labels, rotation=60, ha="right")
    axis.set_title("Per-class diagnostic error delta (negative is better)")
    figure.tight_layout()
    figure.savefig(str(path), dpi=160)
    plt.close(figure)


def _rows_tracks_plot(plt: Any, path: Path, rows: List[Dict[str, Any]]) -> None:
    labels = [str(row.get("variant", "")) for row in rows]
    row_values = [_number(row.get("rows")) for row in rows]
    tracks = [_number(row.get("unique_tracks")) for row in rows]
    figure, axes = plt.subplots(1, 2, figsize=(14, 5))
    for axis, values, title in [(axes[0], row_values, "Rows"), (axes[1], tracks, "Unique tracks")]:
        axis.bar(range(len(labels)), values)
        axis.set_xticks(range(len(labels)))
        axis.set_xticklabels(labels, rotation=30, ha="right")
        axis.set_title(title)
    figure.tight_layout()
    figure.savefig(str(path), dpi=160)
    plt.close(figure)


def _number(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0
