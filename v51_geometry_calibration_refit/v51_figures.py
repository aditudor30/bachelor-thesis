"""Generate requested Step 22E diagnostic figures."""

from pathlib import Path
from typing import Any, Dict, List

from deep_oc_sort_3d.v51_geometry_calibration_refit.v51_config import output_root
from deep_oc_sort_3d.v51_geometry_calibration_refit.v51_io import read_csv


def write_v51_figures(config: Dict[str, Any]) -> List[Path]:
    import matplotlib.pyplot as plt

    root = output_root(config)
    rows = read_csv(root / "validation_diagnostics" / "official_val_before_after.csv")
    directory = root / "figures"
    directory.mkdir(parents=True, exist_ok=True)
    output: List[Path] = []
    for filename, metric, title in [
        ("center_error_before_after.png", "center_error_mean", "Official val center error"),
        ("dimension_error_before_after.png", "dimension_error_mean", "Official val dimension error"),
        ("yaw_error_before_after.png", "yaw_error_mean", "Official val yaw error"),
    ]:
        path = directory / filename
        _before_after(plt, path, [row for row in rows if row.get("metric") == metric], title)
        output.append(path)
    path = directory / "per_class_improvement.png"
    _deltas(plt, path, read_csv(root / "validation_diagnostics" / "per_class_before_after.csv"), "Per-class error delta")
    output.append(path)
    path = directory / "v5_vs_v51_position_change.png"
    _deltas(plt, path, read_csv(root / "validation_diagnostics" / "v5_vs_v51_diagnostics.csv"), "V5 versus V5.1 diagnostics")
    output.append(path)
    return output


def _before_after(plt: Any, path: Path, rows: List[Dict[str, Any]], title: str) -> None:
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
    figure.tight_layout()
    figure.savefig(str(path), dpi=160)
    plt.close(figure)


def _deltas(plt: Any, path: Path, rows: List[Dict[str, Any]], title: str) -> None:
    selected = [row for row in rows if row.get("delta") not in (None, "") or row.get("candidate_value") not in (None, "")]
    labels = [str(row.get("metric", row.get("official_class_id", ""))) for row in selected]
    values = [_number(row.get("delta", row.get("candidate_value"))) for row in selected]
    figure, axis = plt.subplots(figsize=(14, 6))
    axis.bar(range(len(labels)), values, color=["#2878B5" if value <= 0.0 else "#D95319" for value in values])
    axis.axhline(0.0, color="black", linewidth=1)
    axis.set_xticks(range(len(labels)))
    axis.set_xticklabels(labels, rotation=60, ha="right")
    axis.set_title(title)
    figure.tight_layout()
    figure.savefig(str(path), dpi=160)
    plt.close(figure)


def _number(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0
