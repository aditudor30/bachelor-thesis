"""Summarize local tracklet outputs."""

import argparse
import json
from pathlib import Path
from typing import Any, Iterable, List

from deep_oc_sort_3d.tracklets.tracklet_summary import (
    print_tracklet_summary,
    summarize_tracklet_files,
    write_tracklet_summary_csv,
    write_tracklet_summary_json,
)


def summarize_local_tracklets(args: Any) -> None:
    """Print local tracklet summary."""
    summary_json = args.tracklet_root / "summaries" / "tracklet_summary.json"
    if summary_json.exists() and not args.rescan:
        summary = json.loads(summary_json.read_text(encoding="utf-8"))
    else:
        files = _resolve_tracklet_files(args.tracklet_root)
        files = list(_progress_iter(files, args.progress, "summarize local tracklets"))
        summary = summarize_tracklet_files(files)
        if args.write_summary:
            write_tracklet_summary_csv(summary, args.tracklet_root / "summaries" / "tracklet_summary.csv")
            write_tracklet_summary_json(summary, args.tracklet_root / "summaries" / "tracklet_summary.json")
    print_tracklet_summary(summary)


def _resolve_tracklet_files(root: Path) -> List[Path]:
    jsonl_files = sorted(root.rglob("*_tracklets.jsonl"))
    if jsonl_files:
        return jsonl_files
    return sorted(root.rglob("*_tracklets.csv"))


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
    parser = argparse.ArgumentParser(description="Summarize local tracklets.")
    parser.add_argument("--tracklet-root", required=True, type=Path)
    parser.add_argument("--rescan", action="store_true")
    parser.add_argument("--write-summary", action="store_true")
    progress_group = parser.add_mutually_exclusive_group()
    progress_group.add_argument("--progress", dest="progress", action="store_true")
    progress_group.add_argument("--no-progress", dest="progress", action="store_false")
    parser.set_defaults(progress=True)
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    summarize_local_tracklets(args)


if __name__ == "__main__":
    main()
