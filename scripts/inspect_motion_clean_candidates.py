"""Inspect motion-clean candidate output."""

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Iterable, List

from deep_oc_sort_3d.mtmc.candidate_motion_io import read_motion_metrics_csv
from deep_oc_sort_3d.scripts.audit_candidate_motion_quality import print_motion_summary, summarize_motion_metrics


def inspect_motion_clean_candidates(args: Any) -> None:
    """Inspect output from motion filtering."""
    summary_path = args.root / "summaries" / "motion_quality_summary.json"
    if summary_path.exists() and not args.rescan:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    else:
        metrics = []
        files = sorted(args.root.rglob("*_motion_metrics.csv"))
        for path in _progress_iter(files, args.progress, "inspect motion metrics"):
            metrics.extend(read_motion_metrics_csv(path))
        summary = summarize_motion_metrics(metrics)
    print_motion_summary(summary)
    _print_file_counts(args.root)
    worst = args.root / "summaries" / "worst_motion_outliers.csv"
    if worst.exists():
        print("worst outliers:")
        for row in _read_csv(worst)[:10]:
            print("  %s %s max_step=%s reason=%s" % (
                row.get("candidate_id"),
                row.get("class_name"),
                row.get("max_step_distance_3d"),
                row.get("motion_reject_reason"),
            ))


def _print_file_counts(root: Path) -> None:
    clean = len(list(root.rglob("*_clean_candidates.jsonl")))
    suspicious = len(list(root.rglob("*_suspicious_candidates.jsonl")))
    invalid = len(list(root.rglob("*_invalid_candidates.jsonl")))
    unknown = len(list(root.rglob("*_unknown_candidates.jsonl")))
    print("files clean/suspicious/invalid/unknown: %d/%d/%d/%d" % (clean, suspicious, invalid, unknown))


def _read_csv(path: Path):
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _progress_iter(values: List[Any], show_progress: bool, desc: str) -> Iterable[Any]:
    if not show_progress:
        return values
    try:
        from tqdm import tqdm
    except ImportError:
        return _print_progress_iter(values, desc)
    return tqdm(values, desc=desc, unit="file")


def _print_progress_iter(values: List[Any], desc: str) -> Iterable[Any]:
    total = len(values)
    for index, value in enumerate(values):
        print("%s: file %d/%d %s" % (desc, index + 1, total, value))
        yield value


def build_arg_parser() -> argparse.ArgumentParser:
    """Build CLI parser."""
    parser = argparse.ArgumentParser(description="Inspect motion-clean MTMC candidates.")
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--rescan", action="store_true")
    progress_group = parser.add_mutually_exclusive_group()
    progress_group.add_argument("--progress", dest="progress", action="store_true")
    progress_group.add_argument("--no-progress", dest="progress", action="store_false")
    parser.set_defaults(progress=True)
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    inspect_motion_clean_candidates(args)


if __name__ == "__main__":
    main()
