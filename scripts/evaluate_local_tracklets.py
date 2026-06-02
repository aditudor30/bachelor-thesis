"""Evaluate local tracklet files."""

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

from deep_oc_sort_3d.tracklets.tracklet_eval import (
    evaluate_tracklets,
    save_tracklet_eval_csv,
    save_tracklet_eval_json,
)
from deep_oc_sort_3d.tracklets.tracklet_io import read_tracklets_file


def evaluate_local_tracklets(args: Any) -> None:
    """Evaluate one local tracklet file or a directory recursively."""
    files = _resolve_tracklet_files(args.tracklets)
    rows = []
    all_tracklets = []
    for path in _progress_iter(files, args.progress, "evaluate local tracklets"):
        tracklets = read_tracklets_file(path)
        metrics = evaluate_tracklets(tracklets)
        rows.append(_row_from_metrics(path, metrics))
        all_tracklets.extend(tracklets)
    if args.tracklets.is_file():
        metrics = evaluate_tracklets(all_tracklets)
        save_tracklet_eval_json(metrics, args.output)
        if args.csv_output is not None:
            save_tracklet_eval_csv(metrics, args.csv_output)
    else:
        global_metrics = evaluate_tracklets(all_tracklets)
        payload = {"global": global_metrics, "files": rows}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        if args.csv_output is not None:
            _write_rows(rows, args.csv_output)
    print("evaluated files: %d" % len(files))
    print("total tracklets: %d" % len(all_tracklets))
    print("Wrote %s" % args.output)


def _resolve_tracklet_files(path: Path) -> List[Path]:
    if path.is_file():
        return [path]
    jsonl_files = sorted(path.rglob("*_tracklets.jsonl"))
    if jsonl_files:
        return jsonl_files
    return sorted(path.rglob("*_tracklets.csv"))


def _row_from_metrics(path: Path, metrics: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "tracklets": str(path),
        "num_tracklets": metrics.get("num_tracklets"),
        "valid_for_mtmc": metrics.get("valid_for_mtmc"),
        "invalid_count": metrics.get("invalid_count"),
        "mean_length": metrics.get("mean_length"),
        "median_length": metrics.get("median_length"),
        "p25_length": metrics.get("p25_length"),
        "p75_length": metrics.get("p75_length"),
        "mean_confidence": metrics.get("mean_confidence"),
        "gt_available_count": metrics.get("gt_available_count"),
        "purity_mean": metrics.get("purity_mean"),
        "purity_median": metrics.get("purity_median"),
        "mixed_gt_tracklets": metrics.get("mixed_gt_tracklets"),
        "no_3d_tracklets": metrics.get("no_3d_tracklets"),
        "short_tracklets": metrics.get("short_tracklets"),
        "low_confidence_tracklets": metrics.get("low_confidence_tracklets"),
    }


def _write_rows(rows: List[Dict[str, Any]], path: Path) -> None:
    fields = [
        "tracklets",
        "num_tracklets",
        "valid_for_mtmc",
        "invalid_count",
        "mean_length",
        "median_length",
        "p25_length",
        "p75_length",
        "mean_confidence",
        "gt_available_count",
        "purity_mean",
        "purity_median",
        "mixed_gt_tracklets",
        "no_3d_tracklets",
        "short_tracklets",
        "low_confidence_tracklets",
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
    parser = argparse.ArgumentParser(description="Evaluate local tracklets.")
    parser.add_argument("--tracklets", required=True, type=Path)
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
    evaluate_local_tracklets(args)


if __name__ == "__main__":
    main()
