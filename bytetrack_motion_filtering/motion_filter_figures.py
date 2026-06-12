"""Diagnostic figures for the Step 21E comparison."""

from pathlib import Path
from typing import Any, Dict, List

from deep_oc_sort_3d.bytetrack_motion_filtering.motion_filter_config import output_root
from deep_oc_sort_3d.bytetrack_motion_filtering.motion_filter_io import read_csv


def write_motion_filter_figures(config: Dict[str, Any], comparison: Dict[str, Any]) -> List[str]:
    """Write requested figures when matplotlib is available."""
    if not bool(config.get("figures", {}).get("enabled", True)):
        return []
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return []
    root = output_root(config)
    figure_root = root / "figures"
    figure_root.mkdir(parents=True, exist_ok=True)
    rows = comparison.get("rows", [])
    written = []
    written.append(_bar(plt, rows, "motion_clean_retention", "Motion-clean retention", figure_root / "motion_retention_by_variant.png"))
    written.append(_bar(plt, rows, "track1_rows_retention_vs_bytetrack_21c_best", "Track1 retention vs ByteTrack 21C", figure_root / "track1_rows_retention_by_variant.png"))
    written.append(_scatter(plt, rows, "motion_clean_retention", "global_purity_mean", "Purity vs motion retention", figure_root / "purity_vs_retention.png"))
    written.append(_scatter(plt, rows, "motion_clean_retention", "false_merge_rate", "False merges vs motion retention", figure_root / "false_merge_vs_retention.png"))
    gap_rows = read_csv(root / "diagnostics" / "rejected_by_gap_bucket.csv")
    class_rows = read_csv(root / "diagnostics" / "rejected_by_class.csv")
    written.append(_grouped_lines(plt, gap_rows, "gap_bucket", "rate", "Rejection by gap bucket", figure_root / "rejection_by_gap_bucket.png"))
    written.append(_grouped_lines(plt, class_rows, "class_name", "rate", "Rejection by class", figure_root / "rejection_by_class.png"))
    return [value for value in written if value]


def _bar(plt: Any, rows: List[Dict[str, Any]], key: str, title: str, path: Path) -> str:
    names = [str(row.get("variant_name", "")) for row in rows]
    values = [_float(row.get(key)) for row in rows]
    fig, axis = plt.subplots(figsize=(11, 5))
    axis.bar(range(len(names)), values)
    axis.set_xticks(range(len(names)))
    axis.set_xticklabels(names, rotation=25, ha="right")
    axis.set_ylabel(key)
    axis.set_title(title)
    fig.tight_layout()
    fig.savefig(str(path), dpi=160)
    plt.close(fig)
    return str(path)


def _scatter(plt: Any, rows: List[Dict[str, Any]], x_key: str, y_key: str, title: str, path: Path) -> str:
    fig, axis = plt.subplots(figsize=(8, 6))
    for row in rows:
        x_value = _float(row.get(x_key))
        y_value = _float(row.get(y_key))
        axis.scatter([x_value], [y_value])
        axis.annotate(str(row.get("variant_name", "")), (x_value, y_value), fontsize=8)
    axis.set_xlabel(x_key)
    axis.set_ylabel(y_key)
    axis.set_title(title)
    fig.tight_layout()
    fig.savefig(str(path), dpi=160)
    plt.close(fig)
    return str(path)


def _grouped_lines(
    plt: Any,
    rows: List[Dict[str, Any]],
    group_key: str,
    value_key: str,
    title: str,
    path: Path,
) -> str:
    groups = {}
    labels = []
    for row in rows:
        label = str(row.get(group_key, "unknown"))
        if label not in labels:
            labels.append(label)
        groups.setdefault(str(row.get("variant_name", "")), {})[label] = _float(row.get(value_key))
    fig, axis = plt.subplots(figsize=(11, 5))
    for variant, values in groups.items():
        axis.plot(range(len(labels)), [values.get(label, 0.0) for label in labels], marker="o", label=variant)
    axis.set_xticks(range(len(labels)))
    axis.set_xticklabels(labels, rotation=25, ha="right")
    axis.set_ylabel(value_key)
    axis.set_title(title)
    if groups:
        axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(str(path), dpi=160)
    plt.close(fig)
    return str(path)


def _float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0

