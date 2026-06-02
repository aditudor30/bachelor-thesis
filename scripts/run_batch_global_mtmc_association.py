"""Run global MTMC association over multiple scenes."""

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from deep_oc_sort_3d.mtmc.global_associator import GlobalMTMCAssociator
from deep_oc_sort_3d.mtmc.global_summary import summarize_global_association
from deep_oc_sort_3d.scripts.run_global_mtmc_association import (
    _load_config,
    _read_scene_candidates,
    _write_scene_outputs,
)


def run_batch_global_mtmc_association(args: Any) -> None:
    """Run global MTMC association scene-by-scene."""
    config = _load_config(args.config)
    scenes = _find_scene_roots(args.candidates_root, args.subsets, args.scenes)
    rows = []
    for subset, scene_name, scene_root in _progress_iter(scenes, args.progress, "global MTMC scenes", "scene"):
        row = _process_scene(subset, scene_name, scene_root, args, config)
        rows.append(row)
    summary = _aggregate_rows(rows)
    summary_root = args.output_root / "summaries"
    summary_root.mkdir(parents=True, exist_ok=True)
    (summary_root / "global_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    _write_rows_csv(rows, summary_root / "global_summary.csv")
    print("scenes: %d" % len(scenes))
    print("errors: %d" % len([row for row in rows if row.get("status") == "error"]))
    print("global_tracks: %s" % summary.get("global_tracks"))
    print("multi_camera_tracks: %s" % summary.get("multi_camera_tracks"))
    print("Run root: %s" % args.output_root)


def _process_scene(
    subset: str,
    scene_name: str,
    scene_root: Path,
    args: Any,
    config: Dict[str, Any],
) -> Dict[str, Any]:
    output_root = args.output_root / subset / scene_name
    if output_root.exists() and not args.overwrite:
        return _row(subset, scene_name, 0, 0, 0, 0, "skipped_existing", "")
    try:
        candidates = _read_scene_candidates(scene_root, args.class_names, args.max_candidates_per_scene)
        associator = GlobalMTMCAssociator(config=config)
        global_tracks, edges, mapping = associator.associate(candidates, show_progress=args.progress)
        summary = _write_scene_outputs(output_root, candidates, global_tracks, edges, mapping)
        return _row(
            subset,
            scene_name,
            len(candidates),
            len(edges),
            int(summary.get("global_tracks", 0)),
            int(summary.get("multi_camera_tracks", 0)),
            "ok",
            "",
        )
    except Exception as exc:
        return _row(subset, scene_name, 0, 0, 0, 0, "error", str(exc))


def _find_scene_roots(
    candidates_root: Path,
    subsets: Optional[List[str]],
    scenes: Optional[List[str]],
) -> List[Tuple[str, str, Path]]:
    subset_filter = None if subsets is None else set(subsets)
    scene_filter = None if scenes is None else set(scenes)
    output = []
    for subset_dir in sorted(candidates_root.iterdir()):
        if not subset_dir.is_dir() or subset_dir.name == "summaries":
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


def _row(
    subset: str,
    scene_name: str,
    num_candidates: int,
    num_edges: int,
    num_global_tracks: int,
    num_multi_camera_tracks: int,
    status: str,
    error_message: str,
) -> Dict[str, Any]:
    return {
        "subset": subset,
        "scene_name": scene_name,
        "num_candidates": int(num_candidates),
        "num_edges": int(num_edges),
        "num_global_tracks": int(num_global_tracks),
        "num_multi_camera_tracks": int(num_multi_camera_tracks),
        "status": status,
        "error_message": error_message,
    }


def _aggregate_rows(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "scenes": len(rows),
        "ok_scenes": len([row for row in rows if row.get("status") == "ok"]),
        "error_scenes": len([row for row in rows if row.get("status") == "error"]),
        "total_candidates": sum([int(row.get("num_candidates", 0)) for row in rows]),
        "edges": sum([int(row.get("num_edges", 0)) for row in rows]),
        "global_tracks": sum([int(row.get("num_global_tracks", 0)) for row in rows]),
        "multi_camera_tracks": sum([int(row.get("num_multi_camera_tracks", 0)) for row in rows]),
        "files": rows,
    }


def _write_rows_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "subset",
        "scene_name",
        "num_candidates",
        "num_edges",
        "num_global_tracks",
        "num_multi_camera_tracks",
        "status",
        "error_message",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


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
        print("%s: %d/%d %s" % (desc, index + 1, total, value[1]))
        yield value


def build_arg_parser() -> argparse.ArgumentParser:
    """Build CLI parser."""
    parser = argparse.ArgumentParser(description="Batch global MTMC association.")
    parser.add_argument("--candidates-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--subsets", nargs="+", default=None)
    parser.add_argument("--scenes", nargs="+", default=None)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--class-names", nargs="+", default=None)
    parser.add_argument("--max-candidates-per-scene", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    progress_group = parser.add_mutually_exclusive_group()
    progress_group.add_argument("--progress", dest="progress", action="store_true")
    progress_group.add_argument("--no-progress", dest="progress", action="store_false")
    parser.set_defaults(progress=True)
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    run_batch_global_mtmc_association(args)


if __name__ == "__main__":
    main()
