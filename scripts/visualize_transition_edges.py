"""Visualize MTMC transition diagnostics."""

import argparse
from pathlib import Path
from typing import Any, Dict

from deep_oc_sort_3d.mtmc.candidate_io import read_candidates_file
from deep_oc_sort_3d.mtmc.transition_summary import read_transition_pairs_jsonl
from deep_oc_sort_3d.mtmc.transition_visualization import (
    plot_transition_camera_pair_heatmap,
    plot_transition_distance_histogram,
    plot_transition_gap_histogram,
    plot_transition_pairs_bev,
)


def visualize_transition_edges(args: Any) -> None:
    """Create transition diagnostic plots."""
    pairs = read_transition_pairs_jsonl(args.transition_pairs)
    candidates_by_id = _read_candidates_by_id(args.candidates_root)
    accepted = [pair for pair in pairs if pair.accepted_by_threshold]
    near_threshold = [
        pair
        for pair in pairs
        if not pair.accepted_by_threshold and pair.transition_cost is not None and pair.transition_cost < float(args.near_threshold_cost)
    ]
    selected = accepted[: int(args.top_k)] + near_threshold[: int(args.top_k)]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plot_transition_distance_histogram(pairs, args.output_dir / "transition_distance_histogram.png")
    plot_transition_gap_histogram(pairs, args.output_dir / "transition_gap_histogram.png")
    plot_transition_camera_pair_heatmap(pairs, args.output_dir / "transition_camera_pair_heatmap.png")
    plot_transition_pairs_bev(
        candidates_by_id,
        selected,
        args.output_dir / "transition_pairs_bev.png",
        max_pairs=args.top_k * 2,
    )
    print("pairs: %d" % len(pairs))
    print("accepted: %d" % len(accepted))
    print("near_threshold: %d" % len(near_threshold))
    print("output_dir: %s" % args.output_dir)


def _read_candidates_by_id(root: Path) -> Dict[str, Any]:
    files = sorted(root.glob("*_clean_candidates.jsonl"))
    if not files:
        files = sorted(root.glob("*_clean_candidates.csv"))
    if not files:
        files = sorted(root.glob("*_candidates.jsonl"))
    if not files:
        files = sorted(root.glob("*_candidates.csv"))
    output = {}
    for path in files:
        for candidate in read_candidates_file(path):
            output[candidate.candidate_id] = candidate
    return output


def build_arg_parser() -> argparse.ArgumentParser:
    """Build CLI parser."""
    parser = argparse.ArgumentParser(description="Visualize MTMC transition edges.")
    parser.add_argument("--candidates-root", required=True, type=Path)
    parser.add_argument("--transition-pairs", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--near-threshold-cost", type=float, default=1.5)
    progress_group = parser.add_mutually_exclusive_group()
    progress_group.add_argument("--progress", dest="progress", action="store_true")
    progress_group.add_argument("--no-progress", dest="progress", action="store_false")
    parser.set_defaults(progress=True)
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    visualize_transition_edges(args)


if __name__ == "__main__":
    main()
