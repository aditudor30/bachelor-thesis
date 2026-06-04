"""Batch global MTMC association with optional ReID appearance cost."""

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from deep_oc_sort_3d.scripts.run_global_mtmc_with_reid import run_global_mtmc_with_reid


def run_batch_global_mtmc_with_reid(args: Any) -> List[Dict[str, Any]]:
    """Run ReID-aware global MTMC association scene-by-scene."""
    scenes = _find_scene_roots(args.candidates_root, args.subsets, args.scenes)
    rows = []
    for subset, scene_name, scene_root in _progress_iter(scenes, args.progress, "global ReID scenes", "scene"):
        output_root = args.output_root / subset / scene_name
        try:
            scene_args = _SceneArgs(
                candidates_root=scene_root,
                reid_root=args.reid_root,
                output_root=output_root,
                subset=subset,
                scene=scene_name,
                config=args.config,
                appearance_weight=args.appearance_weight,
                use_reid=args.use_reid,
                class_names=args.class_names,
                max_candidates=args.max_candidates_per_scene,
                overwrite=args.overwrite,
                progress=args.progress,
            )
            summary = run_global_mtmc_with_reid(scene_args)
            rows.append(_row(subset, scene_name, output_root, "ok", "", summary))
        except Exception as exc:
            rows.append(_row(subset, scene_name, output_root, "error", str(exc), {}))
    _write_batch_outputs(rows, args.output_root)
    print("scenes: %d" % len(scenes))
    print("errors: %d" % len([row for row in rows if row.get("status") == "error"]))
    print("Run root: %s" % args.output_root)
    return rows


class _SceneArgs:
    """Tiny namespace for forwarding batch scene arguments."""

    def __init__(
        self,
        candidates_root: Path,
        reid_root: Optional[Path],
        output_root: Path,
        subset: str,
        scene: str,
        config: Optional[Path],
        appearance_weight: Optional[float],
        use_reid: Optional[bool],
        class_names: Optional[List[str]],
        max_candidates: Optional[int],
        overwrite: bool,
        progress: bool,
    ) -> None:
        self.candidates_root = candidates_root
        self.reid_root = reid_root
        self.output_root = output_root
        self.subset = subset
        self.scene = scene
        self.config = config
        self.appearance_weight = appearance_weight
        self.use_reid = use_reid
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


def _row(
    subset: str,
    scene_name: str,
    output_root: Path,
    status: str,
    error_message: str,
    summary: Dict[str, Any],
) -> Dict[str, Any]:
    gt_metrics = summary.get("diagnostic_gt_metrics", {}) if isinstance(summary, dict) else {}
    return {
        "subset": subset,
        "scene_name": scene_name,
        "output_root": str(output_root),
        "status": status,
        "error_message": error_message,
        "global_tracks": summary.get("global_tracks", 0) if isinstance(summary, dict) else 0,
        "multi_camera_tracks": summary.get("multi_camera_tracks", 0) if isinstance(summary, dict) else 0,
        "singleton_tracks": summary.get("singleton_tracks", 0) if isinstance(summary, dict) else 0,
        "accepted_edges": summary.get("accepted_edges", 0) if isinstance(summary, dict) else 0,
        "accepted_edges_with_reid": summary.get("accepted_edges_with_reid", 0) if isinstance(summary, dict) else 0,
        "pairs_with_embeddings": summary.get("pairs_with_embeddings", 0) if isinstance(summary, dict) else 0,
        "pairs_missing_embeddings": summary.get("pairs_missing_embeddings", 0) if isinstance(summary, dict) else 0,
        "global_purity_mean": gt_metrics.get("global_purity_mean") if isinstance(gt_metrics, dict) else None,
        "false_merge_rate": gt_metrics.get("false_merge_rate") if isinstance(gt_metrics, dict) else None,
        "fragmentation_approx": gt_metrics.get("fragmentation_approx") if isinstance(gt_metrics, dict) else None,
    }


def _write_batch_outputs(rows: List[Dict[str, Any]], output_root: Path) -> None:
    summary_root = output_root / "summaries"
    summary_root.mkdir(parents=True, exist_ok=True)
    fields = sorted(set([key for row in rows for key in row.keys()])) if rows else ["status"]
    with (summary_root / "global_reid_batch_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    aggregate = _aggregate_rows(rows)
    (summary_root / "global_reid_batch_summary.json").write_text(
        json.dumps({"scenes": rows, "aggregate": aggregate}, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    with (summary_root / "global_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["metric", "value"])
        writer.writeheader()
        for key, value in aggregate.items():
            if isinstance(value, (dict, list)):
                value = json.dumps(value, sort_keys=True)
            writer.writerow({"metric": key, "value": value})
    (summary_root / "global_summary.json").write_text(json.dumps(aggregate, indent=2, sort_keys=True), encoding="utf-8")


def _aggregate_rows(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    ok_rows = [row for row in rows if row.get("status") == "ok"]
    return {
        "files": len(rows),
        "ok": len(ok_rows),
        "errors": len(rows) - len(ok_rows),
        "global_tracks": _sum_int(ok_rows, "global_tracks"),
        "multi_camera_tracks": _sum_int(ok_rows, "multi_camera_tracks"),
        "singleton_tracks": _sum_int(ok_rows, "singleton_tracks"),
        "accepted_edges": _sum_int(ok_rows, "accepted_edges"),
        "accepted_edges_with_reid": _sum_int(ok_rows, "accepted_edges_with_reid"),
        "pairs_with_embeddings": _sum_int(ok_rows, "pairs_with_embeddings"),
        "pairs_missing_embeddings": _sum_int(ok_rows, "pairs_missing_embeddings"),
        "global_purity_mean": _mean([row.get("global_purity_mean") for row in ok_rows]),
        "false_merge_rate": _mean([row.get("false_merge_rate") for row in ok_rows]),
        "fragmentation_approx": _sum_int(ok_rows, "fragmentation_approx"),
    }


def _sum_int(rows: List[Dict[str, Any]], key: str) -> int:
    return int(sum([int(float(row.get(key, 0) or 0)) for row in rows]))


def _mean(values: List[Any]) -> Any:
    filtered = [float(value) for value in values if value not in (None, "")]
    if not filtered:
        return None
    return sum(filtered) / float(len(filtered))


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
    parser = argparse.ArgumentParser(description="Batch global MTMC association with optional ReID cost.")
    parser.add_argument("--candidates-root", required=True, type=Path)
    parser.add_argument("--reid-root", type=Path, default=None)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--subsets", nargs="+", default=None)
    parser.add_argument("--scenes", nargs="+", default=None)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--appearance-weight", type=float, default=None)
    use_group = parser.add_mutually_exclusive_group()
    use_group.add_argument("--use-reid", dest="use_reid", action="store_true")
    use_group.add_argument("--no-use-reid", dest="use_reid", action="store_false")
    parser.set_defaults(use_reid=None)
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
    run_batch_global_mtmc_with_reid(args)


if __name__ == "__main__":
    main()
