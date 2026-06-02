"""Inspect global MTMC association outputs."""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from deep_oc_sort_3d.mtmc.global_io import read_association_edges_jsonl, read_global_tracks_jsonl
from deep_oc_sort_3d.mtmc.global_summary import summarize_global_association


def inspect_global_tracks(args: Any) -> None:
    """Print compact information about global MTMC outputs."""
    summary_files = sorted(args.global_root.rglob("summary.json"))
    if summary_files:
        _print_summaries(summary_files)
        return
    track_files = sorted(args.global_root.rglob("global_tracks.jsonl"))
    edge_files = sorted(args.global_root.rglob("association_edges.jsonl"))
    tracks = []
    edges = []
    for path in track_files:
        tracks.extend(read_global_tracks_jsonl(path))
    for path in edge_files:
        edges.extend(read_association_edges_jsonl(path))
    print("global track files: %d" % len(track_files))
    print("edge files: %d" % len(edge_files))
    print("global tracks: %d" % len(tracks))
    print("multi-camera tracks: %d" % len([track for track in tracks if track.num_cameras > 1]))
    print("singleton tracks: %d" % len([track for track in tracks if track.num_cameras <= 1]))
    print("per class: %s" % json.dumps(_count_by_class(tracks), sort_keys=True))
    print("accepted edges: %d" % len([edge for edge in edges if edge.accepted]))
    print("rejected edges: %d" % len([edge for edge in edges if not edge.accepted]))
    _print_false_merge_examples(tracks, args.top_k)


def _print_summaries(summary_files: List[Path]) -> None:
    total_tracks = 0
    multi_tracks = 0
    singleton_tracks = 0
    accepted_edges = 0
    rejected_edges = 0
    per_class = {}
    for path in summary_files:
        data = json.loads(path.read_text(encoding="utf-8"))
        total_tracks += int(data.get("global_tracks", 0))
        multi_tracks += int(data.get("multi_camera_tracks", 0))
        singleton_tracks += int(data.get("singleton_tracks", 0))
        accepted_edges += int(data.get("accepted_edges", 0))
        rejected_edges += int(data.get("rejected_edges", 0))
        for key, value in data.get("per_class_tracks", {}).items():
            per_class[str(key)] = per_class.get(str(key), 0) + int(value)
    print("summary files: %d" % len(summary_files))
    print("global tracks: %d" % total_tracks)
    print("multi-camera tracks: %d" % multi_tracks)
    print("singleton tracks: %d" % singleton_tracks)
    print("accepted edges: %d" % accepted_edges)
    print("rejected edges: %d" % rejected_edges)
    print("per class: %s" % json.dumps(per_class, sort_keys=True))


def _count_by_class(tracks: List[Any]) -> Dict[str, int]:
    counts = {}
    for track in tracks:
        counts[track.class_name] = counts.get(track.class_name, 0) + 1
    return counts


def _print_false_merge_examples(tracks: List[Any], top_k: int) -> None:
    merges = [track for track in tracks if track.num_gt_ids > 1]
    merges = sorted(merges, key=lambda track: track.num_gt_ids, reverse=True)
    if not merges:
        print("false merge examples: none")
        return
    print("top false merge diagnostics:")
    for track in merges[: int(top_k)]:
        print(
            "  global_track=%d class=%s candidates=%d cameras=%d gt_counts=%s"
            % (
                track.global_track_id,
                track.class_name,
                track.num_candidates,
                track.num_cameras,
                json.dumps(track.gt_id_counts, sort_keys=True),
            )
        )


def build_arg_parser() -> argparse.ArgumentParser:
    """Build CLI parser."""
    parser = argparse.ArgumentParser(description="Inspect global MTMC tracks.")
    parser.add_argument("--global-root", required=True, type=Path)
    parser.add_argument("--top-k", type=int, default=10)
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    inspect_global_tracks(args)


if __name__ == "__main__":
    main()
