"""Apply motion-quality filtering to baseline_v2 MTMC candidates."""

import argparse
from pathlib import Path
from typing import Any, Dict

import yaml

from deep_oc_sort_3d.scripts.filter_mtmc_candidates_by_motion import filter_mtmc_candidates_by_motion


class _Args:
    def __init__(self, cfg: Dict[str, Any], args: Any) -> None:
        section = cfg.get("motion_filtering", cfg)
        self.candidate_root = Path(section.get("candidate_root", "output/mtmc_candidates/baseline_v2_pseudo3d"))
        self.output_root = Path(section.get("output_root", "output/mtmc_candidates_motion_clean/baseline_v2_pseudo3d"))
        self.config = Path(section.get("motion_quality_config", "deep_oc_sort_3d/configs/mtmc_motion_quality_medium_conf001.yaml"))
        self.subsets = section.get("subsets")
        self.scenes = section.get("scenes")
        self.camera_ids = section.get("camera_ids")
        self.allow_suspicious_as_clean = bool(section.get("allow_suspicious_as_clean", False))
        self.require_3d_motion = section.get("require_3d_motion")
        self.overwrite = bool(args.overwrite)
        self.progress = bool(args.progress if args.progress is not None else section.get("progress", True))


def run(args: Any) -> None:
    cfg = _load_yaml(args.config)
    filter_mtmc_candidates_by_motion(_Args(cfg, args))


def _load_yaml(path: Path) -> Dict[str, Any]:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Filter baseline_v2 MTMC candidates by motion quality.")
    parser.add_argument("--config", type=Path, default=Path("deep_oc_sort_3d/configs/baseline_v2_motion_filtering.yaml"))
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

