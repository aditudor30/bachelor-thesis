"""Run transition-enabled global MTMC association for baseline_v2."""

import argparse
from pathlib import Path
from typing import Any, Dict

import yaml

from deep_oc_sort_3d.scripts.run_batch_global_mtmc_with_transitions import run_batch_global_mtmc_with_transitions


class _Args:
    def __init__(self, cfg: Dict[str, Any], args: Any) -> None:
        section = cfg.get("global_association", cfg)
        self.candidates_root = Path(section.get("candidates_root", "output/mtmc_candidates_motion_clean/baseline_v2_pseudo3d"))
        self.output_root = Path(section.get("output_root", "output/global_mtmc_transition/baseline_v2_pseudo3d"))
        self.subsets = section.get("subsets")
        self.scenes = section.get("scenes")
        self.config = Path(section.get("global_config", "deep_oc_sort_3d/configs/global_mtmc_transition_medium_conf001.yaml"))
        self.class_names = section.get("class_names")
        self.max_candidates_per_scene = section.get("max_candidates_per_scene")
        self.overwrite = bool(args.overwrite)
        self.progress = bool(args.progress if args.progress is not None else section.get("progress", True))


def run(args: Any) -> None:
    cfg = _load_yaml(args.config)
    run_batch_global_mtmc_with_transitions(_Args(cfg, args))


def _load_yaml(path: Path) -> Dict[str, Any]:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run baseline_v2 global MTMC association.")
    parser.add_argument("--config", type=Path, default=Path("deep_oc_sort_3d/configs/baseline_v2_global_association.yaml"))
    parser.add_argument("--overwrite", action="store_true")
    progress_group = parser.add_mutually_exclusive_group()
    progress_group.add_argument("--progress", dest="progress", action="store_true", default=None)
    progress_group.add_argument("--no-progress", dest="progress", action="store_false")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()

