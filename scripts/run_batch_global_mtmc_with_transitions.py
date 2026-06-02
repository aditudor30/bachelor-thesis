"""Batch global MTMC association with transition edges."""

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from deep_oc_sort_3d.scripts.run_global_mtmc_with_transitions import (
    run_global_mtmc_with_transitions,
)


def run_batch_global_mtmc_with_transitions(args: Any) -> None:
    """Run transition-enabled global MTMC association scene-by-scene."""
    scenes = _find_scene_roots(args.candidates_root, args.subsets, args.scenes)
    rows = []
    for subset, scene_name, scene_root in _progress_iter(scenes, args.progress, "global transition scenes", "scene"):
        output_root = args.output_root / subset / scene_name
        try:
            scene_args = _SceneArgs(
                candidates_root=scene_root,
                output_root=output_root,
                subset=subset,
                scene=scene_name,
                config=args.config,
                class_names=args.class_names,
                max_candidates=args.max_candidates_per_scene,
                overwrite=args.overwrite,
                progress=args.progress,
            )
            run_global_mtmc_with_transitions(scene_args)
            rows.append(_row(subset, scene_name, output_root, "ok", ""))
        except Exception as exc:
            rows.append(_row(subset, scene_name, output_root, "error", str(exc)))
    summary_root = args.output_root / "summaries"
    summary_root.mkdir(parents=True, exist_ok=True)
    _write_rows_csv(rows, summary_root / "global_transition_summary.csv")
    (summary_root / "global_transition_summary.json").write_text(
        json.dumps({"scenes": rows}, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print("scenes: %d" % len(scenes))
    print("errors: %d" % len([row for row in rows if row.get("status") == "error"]))
    print("Run root: %s" % args.output_root)


class _SceneArgs:
    """Tiny namespace for forwarding batch scene arguments."""

    def __init__(
        self,
        candidates_root: Path,
        output_root: Path,
        subset: str,
        scene: str,
        config: Optional[Path],
        class_names: Optional[List[str]],
        max_candidates: Optional[int],
        overwrite: bool,
        progress: bool,
    ) -> None:
        self.candidates_root = candidates_root
        self.output_root = output_root
        self.subset = subset
        self.scene = scene
        self.config = config
        self.class_names = class_names
        self.max_candidates = max_candidates
        self.overwrite = overwrite
        self.progress = progress


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


def _row(subset: str, scene_name: str, output_root: Path, status: str, error_message: str) -> Dict[str, Any]:
    return {
        "subset": subset,
        "scene_name": scene_name,
        "output_root": str(output_root),
        "status": status,
        "error_message": error_message,
    }


def _write_rows_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    fields = ["subset", "scene_name", "output_root", "status", "error_message"]
    path.parent.mkdir(parents=True, exist_ok=True)
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
        print("%s: %d/%d %s/%s" % (desc, index + 1, total, value[0], value[1]))
        yield value


def build_arg_parser() -> argparse.ArgumentParser:
    """Build CLI parser."""
    parser = argparse.ArgumentParser(description="Batch global MTMC association with transition edges.")
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
    run_batch_global_mtmc_with_transitions(args)


if __name__ == "__main__":
    main()
