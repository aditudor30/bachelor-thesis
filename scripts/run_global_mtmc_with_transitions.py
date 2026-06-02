"""Run global MTMC association with overlap and transition edges."""

import argparse
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from deep_oc_sort_3d.mtmc.global_association_cost import merge_global_association_config
from deep_oc_sort_3d.mtmc.global_association_graph import (
    build_candidate_pairs,
    build_global_tracks_from_edges,
    compute_edges_for_pairs,
)
from deep_oc_sort_3d.mtmc.global_eval import save_global_eval_csv, save_global_eval_json
from deep_oc_sort_3d.mtmc.global_io import (
    write_association_edges_csv,
    write_association_edges_jsonl,
    write_candidates_with_global_ids,
    write_global_tracks_csv,
    write_global_tracks_jsonl,
)
from deep_oc_sort_3d.mtmc.global_summary import (
    print_global_summary,
    summarize_global_association,
    write_global_summary_csv,
    write_global_summary_json,
)
from deep_oc_sort_3d.mtmc.transition_cost import transition_pair_to_global_edge
from deep_oc_sort_3d.mtmc.transition_diagnostics import build_transition_candidate_pairs
from deep_oc_sort_3d.mtmc.transition_summary import write_transition_pairs_csv, write_transition_pairs_jsonl
from deep_oc_sort_3d.scripts.run_global_mtmc_association import _read_scene_candidates


def run_global_mtmc_with_transitions(args: Any) -> None:
    """Run one-scene global association using overlap plus transition edges."""
    config = _resolve_config(args)
    if args.output_root.exists() and not args.overwrite:
        raise ValueError("Output root exists. Pass --overwrite to write: %s" % args.output_root)
    candidates = _read_scene_candidates(args.candidates_root, args.class_names, args.max_candidates)
    if not candidates:
        raise ValueError("No candidates found under %s" % args.candidates_root)
    valid_candidates = [candidate for candidate in candidates if candidate.is_candidate]

    overlap_config = dict(config)
    overlap_config["enable_transition_association"] = False
    overlap_pairs = build_candidate_pairs(valid_candidates, overlap_config, show_progress=args.progress)
    overlap_edges = compute_edges_for_pairs(valid_candidates, overlap_pairs, overlap_config, show_progress=args.progress)

    transition_pairs = build_transition_candidate_pairs(valid_candidates, config, show_progress=args.progress)
    transition_edges = [
        transition_pair_to_global_edge(
            pair,
            float(pair.transition_cost) if pair.transition_cost is not None else 1e9,
            pair.accepted_by_threshold,
            pair.reject_reason,
        )
        for pair in transition_pairs
    ]
    all_edges = overlap_edges + transition_edges
    global_tracks, mapping = build_global_tracks_from_edges(
        valid_candidates,
        all_edges,
        config,
        show_progress=args.progress,
    )
    _write_outputs(args.output_root, candidates, global_tracks, all_edges, transition_edges, transition_pairs, mapping)
    summary = summarize_global_association(global_tracks, all_edges, candidates)
    print_global_summary(summary)
    print("transition_edges: %d" % len(transition_edges))
    print("transition_edges_accepted: %d" % len([edge for edge in transition_edges if edge.accepted]))
    print("Run root: %s" % args.output_root)


def _write_outputs(
    output_root: Path,
    candidates: Any,
    global_tracks: Any,
    all_edges: Any,
    transition_edges: Any,
    transition_pairs: Any,
    mapping: Dict[str, int],
) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    write_global_tracks_csv(global_tracks, output_root / "global_tracks.csv")
    write_global_tracks_jsonl(global_tracks, output_root / "global_tracks.jsonl")
    write_association_edges_csv(all_edges, output_root / "association_edges.csv")
    write_association_edges_jsonl(all_edges, output_root / "association_edges.jsonl")
    write_association_edges_csv(transition_edges, output_root / "transition_edges.csv")
    write_association_edges_jsonl(transition_edges, output_root / "transition_edges.jsonl")
    write_transition_pairs_csv(transition_pairs, output_root / "transition_pairs.csv")
    write_transition_pairs_jsonl(transition_pairs, output_root / "transition_pairs.jsonl")
    write_candidates_with_global_ids(
        candidates,
        mapping,
        output_root / "candidates_with_global_ids.csv",
        output_root / "candidates_with_global_ids.jsonl",
    )
    summary = summarize_global_association(global_tracks, all_edges, candidates)
    write_global_summary_json(summary, output_root / "summary.json")
    write_global_summary_csv(summary, output_root / "summary.csv")
    metrics = summary["diagnostic_gt_metrics"]
    save_global_eval_json(metrics, output_root / "eval.json")
    save_global_eval_csv(metrics, output_root / "eval.csv")


def _resolve_config(args: Any) -> Dict[str, Any]:
    data = _load_config(args.config)
    return merge_global_association_config(data)


def _load_config(path: Optional[Path]) -> Dict[str, Any]:
    if path is None:
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        return {}
    section = data.get("global_mtmc", data)
    return section if isinstance(section, dict) else {}


def build_arg_parser() -> argparse.ArgumentParser:
    """Build CLI parser."""
    parser = argparse.ArgumentParser(description="Run global MTMC association with transition edges.")
    parser.add_argument("--candidates-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--subset", required=True)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--class-names", nargs="+", default=None)
    parser.add_argument("--max-candidates", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    progress_group = parser.add_mutually_exclusive_group()
    progress_group.add_argument("--progress", dest="progress", action="store_true")
    progress_group.add_argument("--no-progress", dest="progress", action="store_false")
    parser.set_defaults(progress=True)
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    run_global_mtmc_with_transitions(args)


if __name__ == "__main__":
    main()
