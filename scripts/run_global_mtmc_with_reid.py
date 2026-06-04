"""Run global MTMC association with optional ReID appearance cost."""

import argparse
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from deep_oc_sort_3d.mtmc.global_eval import save_global_eval_csv, save_global_eval_json
from deep_oc_sort_3d.mtmc.global_io import (
    write_association_edges_csv,
    write_association_edges_jsonl,
    write_candidates_with_global_ids,
    write_global_tracks_csv,
    write_global_tracks_jsonl,
)
from deep_oc_sort_3d.mtmc.global_reid_associator import GlobalMTMCReIDAssociator
from deep_oc_sort_3d.mtmc.global_reid_eval import evaluate_reid_global_tracks
from deep_oc_sort_3d.mtmc.global_reid_summary import (
    print_reid_global_summary,
    summarize_reid_global_association,
    write_reid_global_summary_csv,
    write_reid_global_summary_json,
)
from deep_oc_sort_3d.scripts.run_global_mtmc_association import _read_scene_candidates


def run_global_mtmc_with_reid(args: Any) -> Dict[str, Any]:
    """Run one-scene ReID-aware global MTMC association."""
    config = _resolve_config(args)
    if args.output_root.exists() and not args.overwrite:
        raise ValueError("Output root exists. Pass --overwrite to write: %s" % args.output_root)
    candidates = _read_scene_candidates(args.candidates_root, args.class_names, args.max_candidates)
    if not candidates:
        raise ValueError("No candidates found under %s" % args.candidates_root)
    reid_root = args.reid_root if args.reid_root is not None else _config_reid_root(config)
    associator = GlobalMTMCReIDAssociator(config=config, reid_root=reid_root)
    global_tracks, edges, mapping = associator.associate(candidates, show_progress=args.progress)
    summary = _write_outputs(args.output_root, candidates, global_tracks, edges, mapping)
    print_reid_global_summary(summary)
    print("Run root: %s" % args.output_root)
    return summary


def _write_outputs(output_root: Path, candidates: Any, global_tracks: Any, edges: Any, mapping: Dict[str, int]) -> Dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    write_global_tracks_csv(global_tracks, output_root / "global_tracks.csv")
    write_global_tracks_jsonl(global_tracks, output_root / "global_tracks.jsonl")
    write_association_edges_csv(edges, output_root / "association_edges.csv")
    write_association_edges_jsonl(edges, output_root / "association_edges.jsonl")
    transition_edges = [edge for edge in edges if edge.temporal_relation != "overlap"]
    write_association_edges_csv(transition_edges, output_root / "transition_edges.csv")
    write_association_edges_jsonl(transition_edges, output_root / "transition_edges.jsonl")
    write_candidates_with_global_ids(
        candidates,
        mapping,
        output_root / "candidates_with_global_ids.csv",
        output_root / "candidates_with_global_ids.jsonl",
    )
    summary = summarize_reid_global_association(global_tracks, edges, candidates)
    write_reid_global_summary_json(summary, output_root / "summary.json")
    write_reid_global_summary_csv(summary, output_root / "summary.csv")
    metrics = evaluate_reid_global_tracks(global_tracks, edges)
    save_global_eval_json(metrics, output_root / "eval.json")
    save_global_eval_csv(metrics, output_root / "eval.csv")
    return summary


def _resolve_config(args: Any) -> Dict[str, Any]:
    data = _load_config(args.config)
    if "global_mtmc" not in data:
        data = {"global_mtmc": data, "reid": {}}
    if "reid" not in data or not isinstance(data.get("reid"), dict):
        data["reid"] = {}
    if args.reid_root is not None:
        data["reid"]["reid_root"] = str(args.reid_root)
    if args.appearance_weight is not None:
        data["reid"]["appearance_weight"] = float(args.appearance_weight)
    if args.use_reid is not None:
        data["reid"]["use_reid"] = bool(args.use_reid)
    return data


def _config_reid_root(config: Dict[str, Any]) -> Optional[Path]:
    section = config.get("reid", {})
    if not isinstance(section, dict):
        return None
    value = section.get("reid_root")
    if value in (None, ""):
        return None
    return Path(str(value))


def _load_config(path: Optional[Path]) -> Dict[str, Any]:
    if path is None:
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        return {}
    return data


def build_arg_parser() -> argparse.ArgumentParser:
    """Build CLI parser."""
    parser = argparse.ArgumentParser(description="Run global MTMC association with optional ReID cost.")
    parser.add_argument("--candidates-root", required=True, type=Path)
    parser.add_argument("--reid-root", type=Path, default=None)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--subset", required=True)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--appearance-weight", type=float, default=None)
    use_group = parser.add_mutually_exclusive_group()
    use_group.add_argument("--use-reid", dest="use_reid", action="store_true")
    use_group.add_argument("--no-use-reid", dest="use_reid", action="store_false")
    parser.set_defaults(use_reid=None)
    parser.add_argument("--class-names", nargs="+", default=None)
    parser.add_argument("--max-candidates", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    progress_group = parser.add_mutually_exclusive_group()
    progress_group.add_argument("--progress", dest="progress", action="store_true")
    progress_group.add_argument("--no-progress", dest="progress", action="store_false")
    parser.set_defaults(progress=True)
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    run_global_mtmc_with_reid(args)


if __name__ == "__main__":
    main()
