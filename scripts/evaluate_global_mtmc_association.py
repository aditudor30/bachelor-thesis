"""Evaluate global MTMC association outputs."""

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

from deep_oc_sort_3d.mtmc.global_eval import evaluate_global_tracks
from deep_oc_sort_3d.mtmc.global_io import read_global_tracks_jsonl


def evaluate_global_mtmc_association(args: Any) -> None:
    """Evaluate one folder or tree of global track files."""
    files = _find_global_track_files(args.global_root)
    rows = []
    all_tracks = []
    for path in _progress_iter(files, args.progress, "evaluate global tracks", "file"):
        tracks = read_global_tracks_jsonl(path)
        all_tracks.extend(tracks)
        metrics = evaluate_global_tracks(tracks)
        rows.append(_row(path, metrics))
    aggregate = evaluate_global_tracks(all_tracks)
    aggregate["files"] = rows
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(aggregate, indent=2, sort_keys=True), encoding="utf-8")
    csv_path = args.csv_output if args.csv_output is not None else args.output.with_suffix(".csv")
    _write_rows_csv(rows, csv_path)
    print("files: %d" % len(files))
    print("num_global_tracks: %s" % aggregate.get("num_global_tracks"))
    print("num_multi_camera_tracks: %s" % aggregate.get("num_multi_camera_tracks"))
    print("global_purity_mean: %s" % aggregate.get("global_purity_mean"))


def _find_global_track_files(root: Path) -> List[Path]:
    if root.is_file():
        return [root]
    return sorted(root.rglob("global_tracks.jsonl"))


def _row(path: Path, metrics: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "path": str(path),
        "num_global_tracks": metrics.get("num_global_tracks"),
        "num_multi_camera_tracks": metrics.get("num_multi_camera_tracks"),
        "num_singleton_tracks": metrics.get("num_singleton_tracks"),
        "global_purity_mean": metrics.get("global_purity_mean"),
        "false_merge_rate": metrics.get("false_merge_rate"),
        "fragmentation_approx": metrics.get("fragmentation_approx"),
        "gt_note": metrics.get("gt_note"),
    }


def _write_rows_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "path",
        "num_global_tracks",
        "num_multi_camera_tracks",
        "num_singleton_tracks",
        "global_purity_mean",
        "false_merge_rate",
        "fragmentation_approx",
        "gt_note",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


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
    parser = argparse.ArgumentParser(description="Evaluate global MTMC association.")
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
    evaluate_global_mtmc_association(args)


if __name__ == "__main__":
    main()
