"""Visualize worst MTMC candidate motion outliers."""

import argparse
import csv
from pathlib import Path
from typing import Any, Dict, Iterable, List

from deep_oc_sort_3d.mtmc.candidate_io import read_candidates_file
from deep_oc_sort_3d.mtmc.candidate_motion_io import read_motion_metrics_csv
from deep_oc_sort_3d.mtmc.candidate_motion_visualization import (
    plot_motion_outlier_bev,
    plot_step_distances,
)


def visualize_candidate_motion_outliers(args: Any) -> None:
    """Visualize worst motion outliers from a CSV list."""
    rows = _read_csv(args.metrics)[: int(args.top_k)]
    candidates = _load_candidate_index(args.candidate_root)
    metrics = _load_metrics_index(args.metrics.parent.parent)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for row in _progress_iter(rows, args.progress, "visualize motion outliers"):
        candidate_id = row.get("candidate_id", "")
        candidate = candidates.get(candidate_id)
        metric = metrics.get(candidate_id)
        if candidate is None or metric is None:
            print("warning: missing candidate or metrics for %s" % candidate_id)
            continue
        safe_id = _safe_name(candidate_id)
        plot_motion_outlier_bev(candidate, metric, args.output_dir / ("%s_bev.png" % safe_id))
        plot_step_distances(candidate, metric, args.output_dir / ("%s_steps.png" % safe_id))
    print("Saved outlier visualizations in %s" % args.output_dir)


def _load_candidate_index(root: Path) -> Dict[str, Any]:
    index = {}
    files = sorted(root.rglob("*_candidates.jsonl"))
    if not files:
        files = sorted(root.rglob("*_candidates.csv"))
    for path in files:
        if "summaries" in path.parts:
            continue
        for candidate in read_candidates_file(path):
            index[candidate.candidate_id] = candidate
    return index


def _load_metrics_index(root: Path) -> Dict[str, Any]:
    index = {}
    files = sorted(root.rglob("*_motion_metrics.csv"))
    if not files:
        files = sorted(root.rglob("motion_metrics.csv"))
    for path in files:
        for metrics in read_motion_metrics_csv(path):
            index[metrics.candidate_id] = metrics
    return index


def _read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("_", "-") else "_" for ch in value)


def _progress_iter(values: List[Any], show_progress: bool, desc: str) -> Iterable[Any]:
    if not show_progress:
        return values
    try:
        from tqdm import tqdm
    except ImportError:
        return _print_progress_iter(values, desc)
    return tqdm(values, desc=desc, unit="candidate")


def _print_progress_iter(values: List[Any], desc: str) -> Iterable[Any]:
    total = len(values)
    for index, value in enumerate(values):
        print("%s: candidate %d/%d" % (desc, index + 1, total))
        yield value


def build_arg_parser() -> argparse.ArgumentParser:
    """Build CLI parser."""
    parser = argparse.ArgumentParser(description="Visualize candidate motion outliers.")
    parser.add_argument("--candidate-root", required=True, type=Path)
    parser.add_argument("--metrics", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--top-k", type=int, default=20)
    progress_group = parser.add_mutually_exclusive_group()
    progress_group.add_argument("--progress", dest="progress", action="store_true")
    progress_group.add_argument("--no-progress", dest="progress", action="store_false")
    parser.set_defaults(progress=True)
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    visualize_candidate_motion_outliers(args)


if __name__ == "__main__":
    main()
