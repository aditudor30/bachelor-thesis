"""Visualize global MTMC association outputs."""

import argparse
from pathlib import Path
from typing import Any, Iterable, List, Optional

from deep_oc_sort_3d.mtmc.candidate_io import read_candidates_jsonl
from deep_oc_sort_3d.mtmc.global_io import read_association_edges_jsonl, read_global_tracks_jsonl
from deep_oc_sort_3d.mtmc.global_visualization import (
    plot_association_graph_summary,
    plot_global_tracks_bev,
    visualize_candidate_links_bev,
    visualize_global_track_bev,
)


def visualize_global_tracks(args: Any) -> None:
    """Create global MTMC BEV debug plots."""
    scene_root = args.global_root / args.subset / args.scene
    if not scene_root.exists():
        scene_root = args.global_root
    tracks = read_global_tracks_jsonl(scene_root / "global_tracks.jsonl")
    edges = read_association_edges_jsonl(scene_root / "association_edges.jsonl")
    candidates = _read_candidates(scene_root / "candidates_with_global_ids.jsonl")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plot_global_tracks_bev(tracks, args.output_dir / "global_tracks_bev.png", max_tracks=args.max_tracks)
    plot_association_graph_summary(edges, args.output_dir / "association_edges_summary.png")
    if candidates:
        visualize_candidate_links_bev(
            candidates,
            edges,
            args.output_dir / "candidate_links_bev.png",
            max_edges=args.max_edges,
        )
    if args.global_track_id is not None:
        selected = [track for track in tracks if track.global_track_id == int(args.global_track_id)]
        if selected:
            visualize_global_track_bev(selected[0], args.output_dir / ("global_track_%d_bev.png" % args.global_track_id))
    print("tracks: %d" % len(tracks))
    print("edges: %d" % len(edges))
    print("output_dir: %s" % args.output_dir)


def _read_candidates(path: Path) -> List[Any]:
    if not path.exists():
        return []
    return read_candidates_jsonl(path)


def build_arg_parser() -> argparse.ArgumentParser:
    """Build CLI parser."""
    parser = argparse.ArgumentParser(description="Visualize global MTMC tracks.")
    parser.add_argument("--global-root", required=True, type=Path)
    parser.add_argument("--subset", required=True)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--max-tracks", type=int, default=100)
    parser.add_argument("--max-edges", type=int, default=100)
    parser.add_argument("--global-track-id", type=int, default=None)
    progress_group = parser.add_mutually_exclusive_group()
    progress_group.add_argument("--progress", dest="progress", action="store_true")
    progress_group.add_argument("--no-progress", dest="progress", action="store_false")
    parser.set_defaults(progress=True)
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    visualize_global_tracks(args)


if __name__ == "__main__":
    main()
