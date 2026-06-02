"""Export generic tracking CSVs from frame-level global records."""

import argparse
from pathlib import Path
from typing import Any, Iterable, List, Optional, Tuple

from deep_oc_sort_3d.final_export.export_summary import (
    summarize_generic_exports,
    write_summary_csv,
    write_summary_json,
)
from deep_oc_sort_3d.final_export.generic_export import export_generic_tracking_scene_csv


def export_generic_tracking_results(args: Any) -> None:
    """Export one generic CSV per subset/scene."""
    scenes = _find_scene_roots(args.frame_records_root, args.subsets, args.scenes)
    rows = []
    for subset, scene_name, scene_root in _progress_iter(scenes, args.progress, "generic export scenes", "scene"):
        output_path = args.output_root / subset / ("%s.csv" % scene_name)
        if output_path.exists() and not args.overwrite:
            rows.append({"scene_name": scene_name, "rows_written": 0, "unique_global_tracks": 0, "status": "skipped_existing"})
            continue
        files = sorted(scene_root.glob("*_global_records.csv"))
        row = export_generic_tracking_scene_csv(
            files,
            output_path,
            drop_unassigned=args.drop_unassigned,
            drop_invalid_bbox=args.drop_invalid_bbox,
        )
        row["subset"] = subset
        row["status"] = "ok"
        rows.append(row)
    summary = summarize_generic_exports(rows)
    summary_root = args.output_root.parent / "summaries"
    write_summary_json(summary, summary_root / "export_summary.json")
    write_summary_csv(summary, summary_root / "export_summary.csv")
    print("scenes: %d" % len(scenes))
    print("rows_written: %s" % summary.get("rows_written"))
    print("output_root: %s" % args.output_root)


def _find_scene_roots(root: Path, subsets: Optional[List[str]], scenes: Optional[List[str]]) -> List[Tuple[str, str, Path]]:
    subset_filter = None if subsets is None else set(subsets)
    scene_filter = None if scenes is None else set(scenes)
    output = []
    for subset_dir in sorted(root.iterdir()):
        if not subset_dir.is_dir():
            continue
        if subset_filter is not None and subset_dir.name not in subset_filter:
            continue
        for scene_dir in sorted(subset_dir.iterdir()):
            if not scene_dir.is_dir():
                continue
            if scene_filter is not None and scene_dir.name not in scene_filter:
                continue
            output.append((subset_dir.name, scene_dir.name, scene_dir))
    return output


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
        print("%s: %d/%d %s/%s" % (desc, index + 1, total, value[0], value[1]))
        yield value


def build_arg_parser() -> argparse.ArgumentParser:
    """Build CLI parser."""
    parser = argparse.ArgumentParser(description="Export generic tracking CSVs.")
    parser.add_argument("--frame-records-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--subsets", nargs="+", default=None)
    parser.add_argument("--scenes", nargs="+", default=None)
    assign_group = parser.add_mutually_exclusive_group()
    assign_group.add_argument("--drop-unassigned", dest="drop_unassigned", action="store_true")
    assign_group.add_argument("--include-unassigned", dest="drop_unassigned", action="store_false")
    bbox_group = parser.add_mutually_exclusive_group()
    bbox_group.add_argument("--drop-invalid-bbox", dest="drop_invalid_bbox", action="store_true")
    bbox_group.add_argument("--keep-invalid-bbox", dest="drop_invalid_bbox", action="store_false")
    parser.add_argument("--overwrite", action="store_true")
    progress_group = parser.add_mutually_exclusive_group()
    progress_group.add_argument("--progress", dest="progress", action="store_true")
    progress_group.add_argument("--no-progress", dest="progress", action="store_false")
    parser.set_defaults(drop_unassigned=True, drop_invalid_bbox=True, progress=True)
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    export_generic_tracking_results(args)


if __name__ == "__main__":
    main()
