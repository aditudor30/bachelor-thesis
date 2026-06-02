"""Diagnose MTMC transition candidate pairs."""

import argparse
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from deep_oc_sort_3d.mtmc.transition_cost import merge_transition_config
from deep_oc_sort_3d.mtmc.transition_diagnostics import build_transition_candidate_pairs
from deep_oc_sort_3d.mtmc.transition_summary import (
    print_transition_summary,
    summarize_and_write_transition_pairs,
)
from deep_oc_sort_3d.scripts.run_global_mtmc_association import _read_scene_candidates


def diagnose_mtmc_transitions(args: Any) -> None:
    """Run transition diagnostics for one scene."""
    config = _resolve_config(args)
    if args.output_root.exists() and not args.overwrite:
        raise ValueError("Output root exists. Pass --overwrite to write: %s" % args.output_root)
    candidates = _read_scene_candidates(args.candidates_root, args.class_names, args.max_candidates)
    pairs = build_transition_candidate_pairs(candidates, config, show_progress=args.progress)
    args.output_root.mkdir(parents=True, exist_ok=True)
    summary = summarize_and_write_transition_pairs(pairs, args.output_root)
    print_transition_summary(summary)
    print("Run root: %s" % args.output_root)


def _resolve_config(args: Any) -> Dict[str, Any]:
    data = _load_config(args.config)
    return merge_transition_config(data)


def _load_config(path: Optional[Path]) -> Dict[str, Any]:
    if path is None:
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        return {}
    section = data.get("transition", data)
    return section if isinstance(section, dict) else {}


def build_arg_parser() -> argparse.ArgumentParser:
    """Build CLI parser."""
    parser = argparse.ArgumentParser(description="Diagnose MTMC transition candidate pairs.")
    parser.add_argument("--candidates-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--subset", required=True)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--config", type=Path, default=None)
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
    diagnose_mtmc_transitions(args)


if __name__ == "__main__":
    main()
