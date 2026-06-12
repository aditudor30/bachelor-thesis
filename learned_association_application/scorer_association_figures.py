"""Diagnostic figures for learned association application."""

from pathlib import Path
from typing import Any, Dict, List


def create_figures(scored_rows: List[Dict[str, Any]], comparison_rows: List[Dict[str, Any]], output_root: Path) -> List[str]:
    """Create requested plots when matplotlib is available."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return []
    figures = []
    scores = [float(row.get("mlp_score", 0.0)) for row in scored_rows]
    reid = []
    mlp = []
    for row in scored_rows:
        try:
            reid.append(float(row.get("reid_similarity")))
            mlp.append(float(row.get("mlp_score")))
        except (TypeError, ValueError):
            continue
    figures.append(_histogram(plt, scores, output_root / "mlp_score_distribution.png"))
    figures.append(_scatter(plt, reid, mlp, output_root / "scorer_vs_reid_similarity.png"))
    figures.append(_tradeoff(plt, comparison_rows, "person_false_merge_rate", output_root / "threshold_tradeoff_fragmentation_vs_falsemerge.png"))
    figures.append(_tradeoff(plt, comparison_rows, "person_purity_mean", output_root / "threshold_tradeoff_purity_vs_fragmentation.png"))
    figures.append(_barplot(plt, comparison_rows, output_root / "variant_comparison_barplot.png"))
    return [value for value in figures if value]


def _histogram(plt: Any, values: List[float], path: Path) -> str:
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.hist(values, bins=50)
    axis.set_xlabel("MLP score")
    axis.set_ylabel("Candidate pairs")
    axis.set_title("Person candidate MLP score distribution")
    return _save(plt, figure, path)


def _scatter(plt: Any, x: List[float], y: List[float], path: Path) -> str:
    figure, axis = plt.subplots(figsize=(7, 6))
    axis.scatter(x, y, s=4, alpha=0.25)
    axis.set_xlabel("Fine-tuned ReID similarity")
    axis.set_ylabel("MLP score")
    axis.set_title("Learned scorer vs ReID")
    return _save(plt, figure, path)


def _tradeoff(plt: Any, rows: List[Dict[str, Any]], y_key: str, path: Path) -> str:
    figure, axis = plt.subplots(figsize=(8, 5))
    for row in rows:
        x = row.get("person_fragmentation")
        y = row.get(y_key)
        if x is None or y is None:
            continue
        axis.scatter([float(x)], [float(y)], label=str(row.get("run_name")))
    if axis.has_data():
        axis.legend(fontsize=7)
    axis.set_xlabel("Person fragmentation")
    axis.set_ylabel(y_key)
    return _save(plt, figure, path)


def _barplot(plt: Any, rows: List[Dict[str, Any]], path: Path) -> str:
    figure, axis = plt.subplots(figsize=(10, 5))
    names = [str(row.get("run_name")) for row in rows]
    values = [float(row.get("person_fragmentation") or 0.0) for row in rows]
    axis.bar(range(len(names)), values)
    axis.set_xticks(range(len(names)))
    axis.set_xticklabels(names, rotation=30, ha="right")
    axis.set_ylabel("Person fragmentation")
    figure.tight_layout()
    return _save(plt, figure, path)


def _save(plt: Any, figure: Any, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(str(path), dpi=160)
    plt.close(figure)
    return str(path)
