"""Compare MTMC candidate set summaries."""

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List

from deep_oc_sort_3d.mtmc.candidate_io import read_candidates_file
from deep_oc_sort_3d.mtmc.candidate_summary import summarize_candidates


def compare_mtmc_candidate_sets(args: Any) -> None:
    """Compare multiple candidate roots."""
    names = args.names or [Path(path).name for path in args.sets]
    if len(names) != len(args.sets):
        raise ValueError("--names must match --sets length")
    rows = []
    for name, root in zip(names, args.sets):
        summary = _load_or_scan_summary(root)
        rows.append(_row(name, root, summary))
    _write_rows(rows, args.output)
    print("sets: %d" % len(rows))
    print("Wrote %s" % args.output)


def _load_or_scan_summary(root: Path) -> Dict[str, Any]:
    summary_path = root / "summaries" / "candidate_summary.json"
    if summary_path.exists():
        return json.loads(summary_path.read_text(encoding="utf-8"))
    candidates = []
    files = sorted(root.rglob("*_candidates.jsonl"))
    if not files:
        files = sorted(root.rglob("*_candidates.csv"))
    for path in files:
        candidates.extend(read_candidates_file(path))
    return summarize_candidates(candidates)


def _row(name: str, root: Path, summary: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "name": name,
        "root": str(root),
        "total_candidates": summary.get("total_candidates_including_rejected"),
        "kept_candidates": summary.get("kept_candidates"),
        "kept_ratio": summary.get("kept_ratio"),
        "mean_length": summary.get("mean_length"),
        "candidate_mean_length": summary.get("candidate_mean_length"),
        "has_3d_count": summary.get("has_3d_count"),
        "no_3d_count": summary.get("no_3d_count"),
        "diagnostic_purity_mean": summary.get("diagnostic_purity_mean"),
        "per_class_json": json.dumps(summary.get("per_class_kept_counts", {}), sort_keys=True),
        "per_subset_json": json.dumps(summary.get("per_subset_counts", {}), sort_keys=True),
        "reject_reasons_json": json.dumps(summary.get("reject_reason_counts", {}), sort_keys=True),
    }


def _write_rows(rows: List[Dict[str, Any]], path: Path) -> None:
    fields = [
        "name",
        "root",
        "total_candidates",
        "kept_candidates",
        "kept_ratio",
        "mean_length",
        "candidate_mean_length",
        "has_3d_count",
        "no_3d_count",
        "diagnostic_purity_mean",
        "per_class_json",
        "per_subset_json",
        "reject_reasons_json",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def build_arg_parser() -> argparse.ArgumentParser:
    """Build CLI parser."""
    parser = argparse.ArgumentParser(description="Compare MTMC candidate sets.")
    parser.add_argument("--sets", required=True, nargs="+", type=Path)
    parser.add_argument("--names", nargs="+", default=None)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    compare_mtmc_candidate_sets(args)


if __name__ == "__main__":
    main()
