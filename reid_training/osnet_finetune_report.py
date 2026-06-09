"""Report helpers for OSNet Person ReID fine-tuning outputs."""

import csv
from pathlib import Path
from typing import Any, Dict, List, Optional

from deep_oc_sort_3d.reid_training.reid_dataset_io import read_json, write_json


def summarize_osnet_finetune_output(output_root: Path) -> Dict[str, Any]:
    """Collect the main fine-tuning and evaluation artifacts into one summary."""
    output_root = Path(output_root)
    training_summary = read_json(output_root / "reports" / "training_summary.json") or {}
    evaluation_summary = read_json(output_root / "reports" / "osnet_finetune_evaluation_summary.json") or {}
    environment = read_json(output_root / "logs" / "environment_summary.json") or {}
    log_rows = read_training_log(output_root / "logs" / "training_log.csv")
    summary = {
        "status": "ok" if output_root.exists() else "missing_output_root",
        "output_root": str(output_root),
        "training": training_summary,
        "evaluation": evaluation_summary,
        "environment": environment,
        "num_log_rows": len(log_rows),
        "last_epoch": log_rows[-1].get("epoch") if log_rows else None,
        "best_train_loss": _best_float(log_rows, "train_loss", lower_is_better=True),
        "best_val_top1": _best_float(log_rows, "val_top1_accuracy", lower_is_better=False),
    }
    write_json(summary, output_root / "reports" / "osnet_person_smartspaces_finetune_summary.json")
    write_markdown_report(summary, output_root / "reports" / "osnet_person_smartspaces_finetune_report.md")
    create_report_figures(output_root, log_rows, evaluation_summary)
    return summary


def read_training_log(path: Path) -> List[Dict[str, Any]]:
    """Read training log CSV if present."""
    if not Path(path).exists():
        return []
    with Path(path).open("r", newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def write_markdown_report(summary: Dict[str, Any], path: Path) -> None:
    """Write a compact Markdown report."""
    evaluation = summary.get("evaluation", {})
    verdict = (evaluation.get("verdict") or {}).get("verdict")
    pretrained = ((evaluation.get("pretrained") or {}).get("retrieval") or {})
    finetuned = ((evaluation.get("finetuned") or {}).get("retrieval") or {})
    deltas = evaluation.get("deltas", {})
    lines = [
        "# OSNet Person ReID Fine-Tuning Report",
        "",
        "status: %s" % summary.get("status"),
        "verdict: %s" % verdict,
        "output_root: %s" % summary.get("output_root"),
        "",
        "## Retrieval",
        "",
        "pretrained_top1: %s" % pretrained.get("top1_accuracy"),
        "finetuned_top1: %s" % finetuned.get("top1_accuracy"),
        "top1_delta: %s" % deltas.get("top1_delta"),
        "pretrained_top5: %s" % pretrained.get("top5_accuracy"),
        "finetuned_top5: %s" % finetuned.get("top5_accuracy"),
        "top5_delta: %s" % deltas.get("top5_delta"),
        "mAP_delta: %s" % deltas.get("mAP_delta"),
        "",
        "## Notes",
        "",
        "- This step fine-tunes Person ReID only.",
        "- It does not modify the MTMC pipeline or Track 1 export outputs.",
        "- Use the verdict as a diagnostic before integrating ReID into association.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def create_report_figures(output_root: Path, log_rows: List[Dict[str, Any]], evaluation_summary: Dict[str, Any]) -> Dict[str, Any]:
    """Create lightweight diagnostic figures when matplotlib is available."""
    output_root = Path(output_root)
    figures_dir = output_root / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    created: Dict[str, Any] = {}
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        created["status"] = "matplotlib_unavailable"
        write_json(created, output_root / "figures" / "figure_summary.json")
        return created
    if log_rows:
        path = figures_dir / "training_curves.png"
        _plot_training_curves(plt, log_rows, path)
        created["training_curves"] = str(path)
    if evaluation_summary:
        path = figures_dir / "pretrained_vs_finetuned_retrieval.png"
        _plot_retrieval_comparison(plt, evaluation_summary, path)
        created["retrieval_comparison"] = str(path)
        path = figures_dir / "similarity_margin_summary.png"
        _plot_similarity_summary(plt, evaluation_summary, path)
        created["similarity_summary"] = str(path)
    created["status"] = "ok"
    write_json(created, output_root / "figures" / "figure_summary.json")
    return created


def _plot_training_curves(plt: Any, rows: List[Dict[str, Any]], path: Path) -> None:
    epochs = [_to_float(row.get("epoch"), 0.0) for row in rows]
    train_loss = [_to_float(row.get("train_loss"), 0.0) for row in rows]
    top1 = [_to_float(row.get("val_top1_accuracy"), 0.0) for row in rows]
    fig, axis1 = plt.subplots(figsize=(8, 4))
    axis1.plot(epochs, train_loss, label="train loss", color="tab:blue")
    axis1.set_xlabel("epoch")
    axis1.set_ylabel("loss")
    axis2 = axis1.twinx()
    axis2.plot(epochs, top1, label="val top1", color="tab:orange")
    axis2.set_ylabel("top1")
    axis1.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(str(path), dpi=160)
    plt.close(fig)


def _plot_retrieval_comparison(plt: Any, summary: Dict[str, Any], path: Path) -> None:
    pretrained = ((summary.get("pretrained") or {}).get("retrieval") or {})
    finetuned = ((summary.get("finetuned") or {}).get("retrieval") or {})
    names = ["top1_accuracy", "top5_accuracy", "top10_accuracy", "mAP"]
    x = list(range(len(names)))
    pre = [_to_float(pretrained.get(name), 0.0) for name in names]
    fine = [_to_float(finetuned.get(name), 0.0) for name in names]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar([value - 0.18 for value in x], pre, width=0.36, label="pretrained")
    ax.bar([value + 0.18 for value in x], fine, width=0.36, label="fine-tuned")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha="right")
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(str(path), dpi=160)
    plt.close(fig)


def _plot_similarity_summary(plt: Any, summary: Dict[str, Any], path: Path) -> None:
    pretrained = ((summary.get("pretrained") or {}).get("similarity") or {})
    finetuned = ((summary.get("finetuned") or {}).get("similarity") or {})
    names = ["pre same", "pre diff", "fine same", "fine diff"]
    values = [
        (((pretrained.get("same_gt") or {}).get("mean")) or 0.0),
        (((pretrained.get("different_gt") or {}).get("mean")) or 0.0),
        (((finetuned.get("same_gt") or {}).get("mean")) or 0.0),
        (((finetuned.get("different_gt") or {}).get("mean")) or 0.0),
    ]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(names, [float(value) for value in values], color=["tab:green", "tab:red", "tab:green", "tab:red"])
    ax.set_ylabel("cosine similarity mean")
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(str(path), dpi=160)
    plt.close(fig)


def _best_float(rows: List[Dict[str, Any]], key: str, lower_is_better: bool) -> Optional[float]:
    values = [_to_float(row.get(key), None) for row in rows]
    values = [value for value in values if value is not None]
    if not values:
        return None
    return float(min(values) if lower_is_better else max(values))


def _to_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    if value in (None, ""):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default
