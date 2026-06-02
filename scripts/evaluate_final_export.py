"""Evaluate final frame-level global export with diagnostic GT."""

import argparse
import json
from pathlib import Path
from typing import Any, Iterable, List, Optional, Tuple

from deep_oc_sort_3d.final_export.export_eval import (
    evaluate_global_frame_records,
    save_final_eval_csv,
    save_final_eval_json,
)
from deep_oc_sort_3d.final_export.generic_export import read_global_frame_records_file


def evaluate_final_export(args: Any) -> None:
    """Evaluate final frame records for selected subsets."""
    subset_dirs = _find_subset_dirs(args.frame_records_root, args.subsets)
    all_records = []
    for subset, subset_dir in _progress_iter(subset_dirs, args.progress, "final eval subsets", "subset"):
        records = _read_records_under(subset_dir)
        all_records.extend(records)
        metrics = evaluate_global_frame_records(records)
        save_final_eval_json(metrics, args.output_root / ("%s_eval.json" % subset))
        save_final_eval_csv(metrics, args.output_root / ("%s_eval.csv" % subset))
        print("%s records=%d purity=%s" % (subset, len(records), metrics.get("global_id_purity_mean")))
    global_metrics = evaluate_global_frame_records(all_records)
    save_final_eval_json(global_metrics, args.output_root / "global_eval.json")
    save_final_eval_csv(global_metrics, args.output_root / "global_eval.csv")
    print("global records: %d" % len(all_records))
    print("global purity: %s" % global_metrics.get("global_id_purity_mean"))


def _find_subset_dirs(root: Path, subsets: Optional[List[str]]) -> List[Tuple[str, Path]]:
    subset_filter = None if subsets is None else set(subsets)
    output = []
    for subset_dir in sorted(root.iterdir()):
        if not subset_dir.is_dir():
            continue
        if subset_filter is not None and subset_dir.name not in subset_filter:
            continue
        output.append((subset_dir.name, subset_dir))
    return output


def _read_records_under(root: Path) -> List[Any]:
    records = []
    for path in sorted(root.rglob("*_global_records.csv")):
        records.extend(read_global_frame_records_file(path))
    return records


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
        print("%s: %d/%d %s" % (desc, index + 1, total, value[0]))
        yield value


def build_arg_parser() -> argparse.ArgumentParser:
    """Build CLI parser."""
    parser = argparse.ArgumentParser(description="Evaluate final MVP export.")
    parser.add_argument("--frame-records-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--subsets", nargs="+", default=None)
    progress_group = parser.add_mutually_exclusive_group()
    progress_group.add_argument("--progress", dest="progress", action="store_true")
    progress_group.add_argument("--no-progress", dest="progress", action="store_false")
    parser.set_defaults(progress=True)
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    evaluate_final_export(args)


if __name__ == "__main__":
    main()
