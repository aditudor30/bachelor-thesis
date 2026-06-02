"""Evaluate transition association outputs."""

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

from deep_oc_sort_3d.mtmc.global_eval import evaluate_global_tracks
from deep_oc_sort_3d.mtmc.global_io import read_association_edges_jsonl, read_global_tracks_jsonl


def evaluate_transition_association(args: Any) -> None:
    """Evaluate overlap and transition edges separately."""
    track_files = _find_files(args.global_root, "global_tracks.jsonl")
    edge_files = _find_files(args.global_root, "association_edges.jsonl")
    all_tracks = []
    all_edges = []
    for path in _progress_iter(track_files, args.progress, "transition eval tracks", "file"):
        all_tracks.extend(read_global_tracks_jsonl(path))
    for path in _progress_iter(edge_files, args.progress, "transition eval edges", "file"):
        all_edges.extend(read_association_edges_jsonl(path))
    metrics = evaluate_global_tracks(all_tracks)
    transition_edges = [edge for edge in all_edges if edge.temporal_relation in ("a_before_b", "b_before_a")]
    overlap_edges = [edge for edge in all_edges if edge.temporal_relation == "overlap"]
    output = {
        "num_tracks": len(all_tracks),
        "num_edges": len(all_edges),
        "overlap_edges": len(overlap_edges),
        "overlap_edges_accepted": len([edge for edge in overlap_edges if edge.accepted]),
        "transition_edges": len(transition_edges),
        "transition_edges_accepted": len([edge for edge in transition_edges if edge.accepted]),
        "global_eval": metrics,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True), encoding="utf-8")
    csv_path = args.csv_output if args.csv_output is not None else args.output.with_suffix(".csv")
    _write_metric_csv(output, csv_path)
    print("num_tracks: %d" % len(all_tracks))
    print("transition_edges: %d" % output["transition_edges"])
    print("transition_edges_accepted: %d" % output["transition_edges_accepted"])
    print("multi_camera_tracks: %s" % metrics.get("num_multi_camera_tracks"))
    print("global_purity_mean: %s" % metrics.get("global_purity_mean"))


def _find_files(root: Path, name: str) -> List[Path]:
    if root.is_file():
        return [root]
    return sorted(root.rglob(name))


def _write_metric_csv(metrics: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["metric", "value"])
        writer.writeheader()
        for key, value in metrics.items():
            if isinstance(value, (dict, list)):
                value = json.dumps(value, sort_keys=True)
            writer.writerow({"metric": key, "value": value})


def _progress_iter(values: List[Any], show_progress: bool, desc: str, unit: str) -> Iterable[Any]:
    if not show_progress:
        return values
    try:
        from tqdm import tqdm
    except ImportError:
        return _print_progress_iter(values, desc)
    return tqdm(values, desc=desc, unit=unit)


def _print_progress_iter(values: List[Any], desc: str) -> Iterable[Any]:
    total = len(values)
    for index, value in enumerate(values):
        print("%s: %d/%d %s" % (desc, index + 1, total, value))
        yield value


def build_arg_parser() -> argparse.ArgumentParser:
    """Build CLI parser."""
    parser = argparse.ArgumentParser(description="Evaluate transition association.")
    parser.add_argument("--global-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--csv-output", type=Path, default=None)
    progress_group = parser.add_mutually_exclusive_group()
    progress_group.add_argument("--progress", dest="progress", action="store_true")
    progress_group.add_argument("--no-progress", dest="progress", action="store_false")
    parser.set_defaults(progress=True)
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    evaluate_transition_association(args)


if __name__ == "__main__":
    main()
