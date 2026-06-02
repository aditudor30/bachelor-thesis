"""Evaluate local tracking CSV files with lightweight GT-id diagnostics."""

import argparse
import csv
from pathlib import Path
from typing import Any, Dict, Iterable, List

from deep_oc_sort_3d.tracking.track_eval import evaluate_local_tracks, save_track_eval_csv, save_track_eval_json
from deep_oc_sort_3d.tracking.track_io import read_local_tracks_csv


def evaluate_local_tracking(args: Any) -> None:
    """Evaluate one track CSV or all CSV files under a directory."""
    files = _resolve_track_files(args.tracks)
    rows = []
    for path in _progress_iter(files, not args.no_progress, "evaluate local tracks"):
        metrics = evaluate_local_tracks(read_local_tracks_csv(path))
        rows.append(_row_from_metrics(path, metrics))
        if args.tracks.is_file():
            save_track_eval_json(metrics, args.output)
            if args.csv_output is not None:
                save_track_eval_csv(metrics, args.csv_output)
    if args.tracks.is_dir():
        args.output.parent.mkdir(parents=True, exist_ok=True)
        import json

        args.output.write_text(json.dumps({"files": rows}, indent=2, sort_keys=True), encoding="utf-8")
        if args.csv_output is not None:
            _write_rows(rows, args.csv_output)
    print("evaluated files: %d" % len(files))
    print("Wrote %s" % args.output)


def _resolve_track_files(path: Path) -> List[Path]:
    if path.is_file():
        return [path]
    return sorted(path.rglob("*.csv"))


def _row_from_metrics(path: Path, metrics: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "tracks": str(path),
        "num_records": metrics.get("num_records"),
        "num_tracks": metrics.get("num_tracks"),
        "mean_track_length": metrics.get("mean_track_length"),
        "median_track_length": metrics.get("median_track_length"),
        "num_gt_matched_records": metrics.get("num_gt_matched_records"),
        "id_switches_approx": metrics.get("id_switches_approx"),
        "fragmentations_approx": metrics.get("fragmentations_approx"),
        "purity_mean": metrics.get("purity_mean"),
        "has_gt": metrics.get("has_gt"),
    }


def _write_rows(rows: List[Dict[str, Any]], path: Path) -> None:
    fields = [
        "tracks",
        "num_records",
        "num_tracks",
        "mean_track_length",
        "median_track_length",
        "num_gt_matched_records",
        "id_switches_approx",
        "fragmentations_approx",
        "purity_mean",
        "has_gt",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _progress_iter(values: List[Path], show_progress: bool, desc: str) -> Iterable[Path]:
    if not show_progress:
        return values
    try:
        from tqdm import tqdm
    except ImportError:
        return _print_progress_iter(values, desc)
    return tqdm(values, desc=desc, unit="file")


def _print_progress_iter(values: List[Path], desc: str) -> Iterable[Path]:
    total = len(values)
    for index, value in enumerate(values):
        print("%s: file %d/%d %s" % (desc, index + 1, total, value))
        yield value


def build_arg_parser() -> argparse.ArgumentParser:
    """Build CLI parser."""
    parser = argparse.ArgumentParser(description="Evaluate local tracking output.")
    parser.add_argument("--tracks", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--csv-output", type=Path, default=None)
    parser.add_argument("--no-progress", action="store_true")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    evaluate_local_tracking(args)


if __name__ == "__main__":
    main()
