"""Inspect MTMC candidate files."""

import argparse
from pathlib import Path
from typing import Any, Iterable, List

from deep_oc_sort_3d.mtmc.candidate_io import read_candidates_file
from deep_oc_sort_3d.mtmc.candidate_summary import print_candidate_summary, summarize_candidates


def inspect_mtmc_candidates(args: Any) -> None:
    """Inspect one candidate file or a folder recursively."""
    files = _resolve_candidate_files(args.candidates)
    all_candidates = []
    for path in _progress_iter(files, args.progress, "inspect MTMC candidates"):
        all_candidates.extend(read_candidates_file(path))
    summary = summarize_candidates(all_candidates)
    print("files: %d" % len(files))
    print_candidate_summary(summary)


def _resolve_candidate_files(path: Path) -> List[Path]:
    if path.is_file():
        return [path]
    jsonl_files = sorted(path.rglob("*_candidates.jsonl"))
    if jsonl_files:
        return jsonl_files
    return sorted(path.rglob("*_candidates.csv"))


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
    parser = argparse.ArgumentParser(description="Inspect MTMC candidates.")
    parser.add_argument("--candidates", required=True, type=Path)
    progress_group = parser.add_mutually_exclusive_group()
    progress_group.add_argument("--progress", dest="progress", action="store_true")
    progress_group.add_argument("--no-progress", dest="progress", action="store_false")
    parser.set_defaults(progress=True)
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    inspect_mtmc_candidates(args)


if __name__ == "__main__":
    main()
