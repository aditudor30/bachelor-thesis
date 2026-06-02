"""Compare multiple local tracklet runs."""

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List

from deep_oc_sort_3d.tracklets.tracklet_summary import summarize_tracklet_files


def compare_tracklet_runs(args: Any) -> None:
    """Compare local tracklet run summaries."""
    names = args.names or [Path(path).name for path in args.runs]
    if len(names) != len(args.runs):
        raise ValueError("--names must match --runs length")
    rows = []
    for name, run_root in zip(names, args.runs):
        summary = _load_or_scan_summary(run_root)
        rows.append(_row(name, run_root, summary))
    _write_rows(rows, args.output)
    print("runs: %d" % len(rows))
    print("Wrote %s" % args.output)


def _load_or_scan_summary(run_root: Path) -> Dict[str, Any]:
    summary_path = run_root / "summaries" / "tracklet_summary.json"
    if summary_path.exists():
        return json.loads(summary_path.read_text(encoding="utf-8"))
    files = sorted(run_root.rglob("*_tracklets.jsonl"))
    if not files:
        files = sorted(run_root.rglob("*_tracklets.csv"))
    return summarize_tracklet_files(files)


def _row(name: str, run_root: Path, summary: Dict[str, Any]) -> Dict[str, Any]:
    quality_flags = summary.get("quality_flags", {})
    return {
        "name": name,
        "run_root": str(run_root),
        "total_files": summary.get("total_files"),
        "total_tracklets": summary.get("total_tracklets"),
        "valid_tracklets": summary.get("valid_tracklets"),
        "mean_length": summary.get("mean_length"),
        "median_length": summary.get("median_length"),
        "purity_mean": summary.get("purity_mean"),
        "no_3d_tracklets": summary.get("no_3d_tracklets"),
        "short_tracklets": quality_flags.get("short", 0),
        "low_confidence_tracklets": quality_flags.get("low_confidence", 0),
        "quality_flags_json": json.dumps(quality_flags, sort_keys=True),
        "per_class_json": json.dumps(summary.get("per_class", {}), sort_keys=True),
    }


def _write_rows(rows: List[Dict[str, Any]], path: Path) -> None:
    fields = [
        "name",
        "run_root",
        "total_files",
        "total_tracklets",
        "valid_tracklets",
        "mean_length",
        "median_length",
        "purity_mean",
        "no_3d_tracklets",
        "short_tracklets",
        "low_confidence_tracklets",
        "quality_flags_json",
        "per_class_json",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def build_arg_parser() -> argparse.ArgumentParser:
    """Build CLI parser."""
    parser = argparse.ArgumentParser(description="Compare local tracklet runs.")
    parser.add_argument("--runs", required=True, nargs="+", type=Path)
    parser.add_argument("--names", nargs="+", default=None)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    compare_tracklet_runs(args)


if __name__ == "__main__":
    main()
