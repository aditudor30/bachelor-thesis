"""Diagnostic figures for the ByteTrack coverage sweep."""

from pathlib import Path
from typing import Any, Dict, List


def write_tuning_figures(
    summary_rows: List[Dict[str, Any]],
    per_class_rows: List[Dict[str, Any]],
    output_root: Path,
) -> None:
    """Write requested plots when matplotlib is available."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    output_root.mkdir(parents=True, exist_ok=True)
    completed = [row for row in summary_rows if row.get("track1_rows") not in (None, 0, "0")]
    _scatter(
        plt,
        completed,
        "track1_rows_retention",
        "global_purity_mean",
        "Track1 retention vs global purity",
        output_root / "retention_vs_purity.png",
    )
    _scatter(
        plt,
        completed,
        "track1_rows_retention",
        "global_fragmentation",
        "Track1 retention vs global fragmentation",
        output_root / "retention_vs_fragmentation.png",
    )
    _bar(plt, completed, "track1_rows_retention", "Track1 row retention", output_root / "track1_rows_retention_by_variant.png")
    _bar(
        plt,
        completed,
        "multi_camera_tracks_retention",
        "Multi-camera track retention",
        output_root / "multi_camera_retention_by_variant.png",
    )
    _bar(plt, summary_rows, "short_track_ratio_le3", "Short-track ratio <= 3", output_root / "short_track_ratio_by_variant.png")
    _per_class_bar(plt, per_class_rows, output_root / "per_class_retention_barplot.png")


def _scatter(plt: Any, rows: List[Dict[str, Any]], x_key: str, y_key: str, title: str, path: Path) -> None:
    figure, axis = plt.subplots(figsize=(8, 5))
    for row in rows:
        x_value = row.get(x_key)
        y_value = row.get(y_key)
        if x_value is None or y_value is None:
            continue
        axis.scatter(float(x_value), float(y_value), label=str(row.get("variant")))
    axis.set_xlabel(x_key)
    axis.set_ylabel(y_key)
    axis.set_title(title)
    if axis.collections:
        axis.legend(fontsize=7)
    figure.tight_layout()
    figure.savefig(str(path), dpi=150)
    plt.close(figure)


def _bar(plt: Any, rows: List[Dict[str, Any]], key: str, title: str, path: Path) -> None:
    labels = [str(row.get("variant")) for row in rows if row.get(key) is not None]
    values = [float(row.get(key)) for row in rows if row.get(key) is not None]
    figure, axis = plt.subplots(figsize=(10, 5))
    axis.bar(range(len(values)), values)
    axis.set_xticks(range(len(labels)))
    axis.set_xticklabels(labels, rotation=35, ha="right")
    axis.set_title(title)
    figure.tight_layout()
    figure.savefig(str(path), dpi=150)
    plt.close(figure)


def _per_class_bar(plt: Any, rows: List[Dict[str, Any]], path: Path) -> None:
    filtered = [
        row for row in rows
        if row.get("stage") == "local_records" and row.get("retention_vs_v2_current") is not None
    ]
    labels = ["%s/%s" % (row.get("variant"), row.get("key")) for row in filtered]
    values = [float(row.get("retention_vs_v2_current")) for row in filtered]
    figure, axis = plt.subplots(figsize=(14, 6))
    axis.bar(range(len(values)), values)
    axis.set_xticks(range(len(labels)))
    axis.set_xticklabels(labels, rotation=75, ha="right", fontsize=7)
    axis.set_title("Per-class local-record retention vs V2 current")
    figure.tight_layout()
    figure.savefig(str(path), dpi=150)
    plt.close(figure)

