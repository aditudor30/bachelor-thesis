"""Figures for Step 18D visual decision diagnostics."""

from pathlib import Path
from typing import Any, Dict, List

from deep_oc_sort_3d.reid_visual_decision.visual_decision_io import safe_float, write_json


def create_visual_decision_figures(rows: List[Dict[str, Any]], output_root: Path) -> Dict[str, Any]:
    """Create lightweight visual-decision summary figures."""
    figures_dir = Path(output_root) / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        summary = {"status": "matplotlib_unavailable"}
        write_json(summary, figures_dir / "figure_summary.json")
        return summary
    summary: Dict[str, Any] = {"status": "ok"}
    if rows:
        path = figures_dir / "reid_similarity_by_auto_label.png"
        plot_similarity_by_label(plt, rows, path)
        summary["reid_similarity_by_auto_label"] = str(path)
        path = figures_dir / "auto_label_counts.png"
        plot_label_counts(plt, rows, path)
        summary["auto_label_counts"] = str(path)
    write_json(summary, figures_dir / "figure_summary.json")
    return summary


def plot_similarity_by_label(plt: Any, rows: List[Dict[str, Any]], path: Path) -> None:
    """Plot ReID similarity grouped by auto label."""
    labels = sorted(set([str(row.get("auto_label", "")) for row in rows]))
    data = []
    for label in labels:
        values = [safe_float(row.get("reid_similarity"), None) for row in rows if str(row.get("auto_label", "")) == label]
        data.append([value for value in values if value is not None])
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.boxplot(data, labels=labels, showfliers=False)
    ax.set_ylabel("ReID cosine similarity")
    ax.tick_params(axis="x", rotation=25)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(str(path), dpi=160)
    plt.close(fig)


def plot_label_counts(plt: Any, rows: List[Dict[str, Any]], path: Path) -> None:
    """Plot auto label counts."""
    counts: Dict[str, int] = {}
    for row in rows:
        label = str(row.get("auto_label", "unknown"))
        counts[label] = counts.get(label, 0) + 1
    labels = sorted(counts.keys())
    values = [counts[label] for label in labels]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(labels, values, color="tab:blue")
    ax.set_ylabel("events")
    ax.tick_params(axis="x", rotation=25)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(str(path), dpi=160)
    plt.close(fig)

