"""Figures for fine-tuned Person ReID association diagnostics."""

from pathlib import Path
from typing import Any, Dict, List

from deep_oc_sort_3d.reid_finetuned_association.finetuned_association_io import read_csv_rows, read_json, safe_float, write_json


def create_finetuned_association_figures(output_root: Path) -> Dict[str, Any]:
    """Create lightweight diagnostic figures."""
    output_root = Path(output_root)
    figures_dir = output_root / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        summary = {"status": "matplotlib_unavailable"}
        write_json(summary, figures_dir / "figure_summary.json")
        return summary
    created: Dict[str, Any] = {"status": "ok"}
    score_rows, _fields = read_csv_rows(output_root / "diagnostics" / "candidate_pair_reid_scores.csv")
    if score_rows:
        path = figures_dir / "reid_score_distribution.png"
        plot_score_distribution(plt, score_rows, path)
        created["reid_score_distribution"] = str(path)
    sweep_rows, _fields = read_csv_rows(output_root / "comparison" / "sweep_summary.csv")
    if sweep_rows:
        path = figures_dir / "threshold_tradeoff_fragmentation_vs_falsemerge.png"
        plot_tradeoff(plt, sweep_rows, path, y_key="false_merge_rate", label="false merge rate")
        created["threshold_tradeoff_fragmentation_vs_falsemerge"] = str(path)
        path = figures_dir / "threshold_tradeoff_purity_vs_fragmentation.png"
        plot_tradeoff(plt, sweep_rows, path, y_key="global_purity_mean", label="purity")
        created["threshold_tradeoff_purity_vs_fragmentation"] = str(path)
        path = figures_dir / "selected_variant_comparison.png"
        plot_variant_bars(plt, sweep_rows, path)
        created["selected_variant_comparison"] = str(path)
    write_json(created, figures_dir / "figure_summary.json")
    return created


def plot_score_distribution(plt: Any, rows: List[Dict[str, Any]], path: Path) -> None:
    """Plot ReID similarity histogram."""
    values = [safe_float(row.get("reid_similarity"), None) for row in rows]
    values = [value for value in values if value is not None]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(values, bins=60, color="tab:blue", alpha=0.8)
    ax.set_xlabel("cosine similarity")
    ax.set_ylabel("candidate pairs")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(str(path), dpi=160)
    plt.close(fig)


def plot_tradeoff(plt: Any, rows: List[Dict[str, Any]], path: Path, y_key: str, label: str) -> None:
    """Plot fragmentation delta against another metric."""
    names = [str(row.get("run_name", "")) for row in rows]
    x_values = [safe_float(row.get("person_fragmentation_delta"), 0.0) or 0.0 for row in rows]
    y_values = [safe_float(row.get(y_key), 0.0) or 0.0 for row in rows]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(x_values, y_values)
    for index, name in enumerate(names):
        ax.annotate(name, (x_values[index], y_values[index]), fontsize=8)
    ax.set_xlabel("person fragmentation delta vs V2")
    ax.set_ylabel(label)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(str(path), dpi=160)
    plt.close(fig)


def plot_variant_bars(plt: Any, rows: List[Dict[str, Any]], path: Path) -> None:
    """Plot a compact variant comparison."""
    names = [str(row.get("run_name", "")) for row in rows]
    frag = [safe_float(row.get("person_fragmentation_delta"), 0.0) or 0.0 for row in rows]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(range(len(names)), frag, color="tab:orange")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=30, ha="right")
    ax.set_ylabel("person fragmentation delta")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(str(path), dpi=160)
    plt.close(fig)
