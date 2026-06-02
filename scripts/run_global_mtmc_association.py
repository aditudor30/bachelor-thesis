"""Run global MTMC association for one subset/scene."""

import argparse
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import yaml

from deep_oc_sort_3d.mtmc.candidate_io import read_candidates_file
from deep_oc_sort_3d.mtmc.candidate_types import MTMCTrackletCandidate
from deep_oc_sort_3d.mtmc.global_association_cost import merge_global_association_config
from deep_oc_sort_3d.mtmc.global_associator import GlobalMTMCAssociator
from deep_oc_sort_3d.mtmc.global_eval import save_global_eval_csv, save_global_eval_json
from deep_oc_sort_3d.mtmc.global_io import (
    write_association_edges_csv,
    write_association_edges_jsonl,
    write_candidates_with_global_ids,
    write_global_tracks_csv,
    write_global_tracks_jsonl,
)
from deep_oc_sort_3d.mtmc.global_summary import (
    print_global_summary,
    summarize_global_association,
    write_global_summary_csv,
    write_global_summary_json,
)


def run_global_mtmc_association(args: Any) -> None:
    """Run global MTMC association for one scene folder."""
    config = _resolve_config(args)
    candidates = _read_scene_candidates(args.candidates_root, args.class_names, args.max_candidates)
    if not candidates:
        raise ValueError("No candidates found under %s" % args.candidates_root)
    if args.output_root.exists() and not args.overwrite:
        raise ValueError("Output root exists. Pass --overwrite to write: %s" % args.output_root)
    associator = GlobalMTMCAssociator(config=config)
    global_tracks, edges, mapping = associator.associate(candidates, show_progress=args.progress)
    _write_scene_outputs(args.output_root, candidates, global_tracks, edges, mapping)
    summary = summarize_global_association(global_tracks, edges, candidates)
    print_global_summary(summary)
    print("Run root: %s" % args.output_root)


def _write_scene_outputs(
    output_root: Path,
    candidates: List[MTMCTrackletCandidate],
    global_tracks: Any,
    edges: Any,
    mapping: Dict[str, int],
) -> Dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    write_global_tracks_csv(global_tracks, output_root / "global_tracks.csv")
    write_global_tracks_jsonl(global_tracks, output_root / "global_tracks.jsonl")
    write_association_edges_csv(edges, output_root / "association_edges.csv")
    write_association_edges_jsonl(edges, output_root / "association_edges.jsonl")
    write_candidates_with_global_ids(
        candidates,
        mapping,
        output_root / "candidates_with_global_ids.csv",
        output_root / "candidates_with_global_ids.jsonl",
    )
    summary = summarize_global_association(global_tracks, edges, candidates)
    write_global_summary_json(summary, output_root / "summary.json")
    write_global_summary_csv(summary, output_root / "summary.csv")
    metrics = summary["diagnostic_gt_metrics"]
    save_global_eval_json(metrics, output_root / "eval.json")
    save_global_eval_csv(metrics, output_root / "eval.csv")
    return summary


def _read_scene_candidates(
    candidates_root: Path,
    class_names: Optional[List[str]] = None,
    max_candidates: Optional[int] = None,
) -> List[MTMCTrackletCandidate]:
    files = _find_candidate_files(candidates_root)
    selected_classes = None if class_names is None else set(class_names)
    candidates = []
    for path in files:
        for candidate in read_candidates_file(path):
            if selected_classes is not None and candidate.class_name not in selected_classes:
                continue
            candidates.append(candidate)
            if max_candidates is not None and len(candidates) >= int(max_candidates):
                return candidates
    return candidates


def _find_candidate_files(candidates_root: Path) -> List[Path]:
    files = sorted(candidates_root.glob("*_clean_candidates.jsonl"))
    if not files:
        files = sorted(candidates_root.glob("*_clean_candidates.csv"))
    if not files:
        files = sorted(candidates_root.glob("*_candidates.jsonl"))
    if not files:
        files = sorted(candidates_root.glob("*_candidates.csv"))
    return files


def _resolve_config(args: Any) -> Dict[str, Any]:
    data = _load_config(args.config)
    if args.max_candidates_per_group is not None:
        data["max_candidates_per_group"] = int(args.max_candidates_per_group)
    return merge_global_association_config(data)


def _load_config(path: Optional[Path]) -> Dict[str, Any]:
    if path is None:
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        return {}
    section = data.get("global_mtmc", data)
    return section if isinstance(section, dict) else {}


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
        if index == 0 or (index + 1) % 10 == 0 or index + 1 == total:
            print("%s: %d/%d" % (desc, index + 1, total))
        yield value


def build_arg_parser() -> argparse.ArgumentParser:
    """Build CLI parser."""
    parser = argparse.ArgumentParser(description="Run global MTMC association for one scene.")
    parser.add_argument("--candidates-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--subset", required=True)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--class-names", nargs="+", default=None)
    parser.add_argument("--max-candidates", type=int, default=None)
    parser.add_argument("--max-candidates-per-group", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    progress_group = parser.add_mutually_exclusive_group()
    progress_group.add_argument("--progress", dest="progress", action="store_true")
    progress_group.add_argument("--no-progress", dest="progress", action="store_false")
    parser.set_defaults(progress=True)
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    run_global_mtmc_association(args)


if __name__ == "__main__":
    main()
