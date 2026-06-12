"""Figures for local tracker benchmark diagnostics."""

from pathlib import Path
from typing import Any, Dict, List, Optional


def create_benchmark_figures(
    rows: List[Dict[str, Any]],
    output_root: Path,
    per_class_rows: Optional[List[Dict[str, Any]]] = None,
) -> List[str]:
    """Create comparison plots when matplotlib is available."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return []
    figures = []
    figures.append(_bar(plt, rows, "mean_track_length", "Mean track length", output_root / "track_length_distribution.png"))
    figures.append(_bar(plt, rows, "short_track_ratio_le3", "Track ratio length <= 3", output_root / "short_track_ratio_by_tracker.png"))
    figures.append(_bar(plt, rows, "person_median_track_length", "Person median track length", output_root / "person_track_length_by_tracker.png"))
    figures.append(_per_class_bar(plt, per_class_rows or [], output_root / "per_class_track_counts.png"))
    figures.append(_bar(plt, rows, "runtime_seconds", "Runtime seconds", output_root / "runtime_by_tracker.png"))
    return figures


def _bar(plt: Any, rows: List[Dict[str, Any]], key: str, ylabel: str, path: Path) -> str:
    names = [str(row.get("tracker_name")) for row in rows if row.get("status") == "ok"]
    values = [float(row.get(key) or 0.0) for row in rows if row.get("status") == "ok"]
    figure, axis = plt.subplots(figsize=(10, 5))
    axis.bar(range(len(names)), values)
    axis.set_xticks(range(len(names)))
    axis.set_xticklabels(names, rotation=30, ha="right")
    axis.set_ylabel(ylabel)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(str(path), dpi=160)
    plt.close(figure)
    return str(path)


def _per_class_bar(plt: Any, rows: List[Dict[str, Any]], path: Path) -> str:
    trackers = sorted(set(str(row.get("tracker_name", "")) for row in rows))
    classes = sorted(
        set(str(row.get("class_name", row.get("class_id", ""))) for row in rows)
    )
    figure, axis = plt.subplots(figsize=(12, 6))
    if trackers and classes:
        width = 0.8 / float(max(1, len(trackers)))
        positions = list(range(len(classes)))
        for tracker_index, tracker_name in enumerate(trackers):
            lookup = {
                str(row.get("class_name", row.get("class_id", ""))): float(row.get("num_tracks") or 0.0)
                for row in rows
                if str(row.get("tracker_name", "")) == tracker_name
            }
            offsets = [position - 0.4 + width / 2.0 + tracker_index * width for position in positions]
            axis.bar(offsets, [lookup.get(name, 0.0) for name in classes], width=width, label=tracker_name)
        axis.set_xticks(positions)
        axis.set_xticklabels(classes, rotation=30, ha="right")
        axis.legend(fontsize=8)
    axis.set_ylabel("Track count")
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(str(path), dpi=160)
    plt.close(figure)
    return str(path)
