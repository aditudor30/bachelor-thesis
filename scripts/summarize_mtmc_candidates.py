"""Summarize MTMC candidate outputs."""

import argparse
import json
from pathlib import Path
from typing import Any, Iterable, List

from deep_oc_sort_3d.mtmc.candidate_io import read_candidates_file
from deep_oc_sort_3d.mtmc.candidate_summary import (
    print_candidate_summary,
    summarize_candidates,
    write_candidate_summary_csv,
    write_candidate_summary_json,
)


def summarize_mtmc_candidates(args: Any) -> None:
    """Summarize candidate root or stored summary."""
    summary_path = args.candidate_root / "summaries" / "candidate_summary.json"
    if summary_path.exists() and not args.rescan:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    else:
        files = _resolve_candidate_files(args.candidate_root)
        candidates = []
        for path in _progress_iter(files, args.progress, "summarize MTMC candidates"):
            candidates.extend(read_candidates_file(path))
        summary = summarize_candidates(candidates)
    print_candidate_summary(summary)
    if args.output_json is not None:
        write_candidate_summary_json(summary, args.output_json)
    if args.output_csv is not None:
        write_candidate_summary_csv(summary, args.output_csv)


def _resolve_candidate_files(root: Path) -> List[Path]:
    jsonl_files = sorted(root.rglob("*_candidates.jsonl"))
    if jsonl_files:
        return jsonl_files
    return sorted(root.rglob("*_candidates.csv"))


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
    parser = argparse.ArgumentParser(description="Summarize MTMC candidates.")
    parser.add_argument("--candidate-root", required=True, type=Path)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--output-csv", type=Path, default=None)
    parser.add_argument("--rescan", action="store_true")
    progress_group = parser.add_mutually_exclusive_group()
    progress_group.add_argument("--progress", dest="progress", action="store_true")
    progress_group.add_argument("--no-progress", dest="progress", action="store_false")
    parser.set_defaults(progress=True)
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    summarize_mtmc_candidates(args)


if __name__ == "__main__":
    main()
