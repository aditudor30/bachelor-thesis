"""Build baseline_v2 Observation3D files enriched with stabilized pseudo-3D."""

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from deep_oc_sort_3d.pseudo3d_integration.pseudo3d_observation_builder import build_pseudo3d_observations_batch


def run(args: Any) -> Dict[str, Any]:
    """Build pseudo-3D observations from a config."""
    cfg = _load_yaml(args.config)
    paths = cfg.get("paths", {})
    subsets_cfg = cfg.get("subsets", {})
    subset_names = args.subsets if args.subsets is not None else list(subsets_cfg.keys())
    scenes = args.scenes
    if scenes is None:
        scenes = _scenes_for_subsets(subsets_cfg, subset_names)
    progress = bool(args.progress if args.progress is not None else cfg.get("progress", True))
    summary = build_pseudo3d_observations_batch(
        input_observations_root=Path(paths.get("input_observations_root")),
        pseudo3d_predictions_root=Path(paths.get("pseudo3d_predictions_root")),
        output_observations_root=Path(paths.get("output_observations_root")),
        config=cfg,
        subsets=subset_names,
        scenes=scenes,
        camera_ids=args.camera_ids,
        show_progress=progress,
        overwrite=bool(args.overwrite),
    )
    print("pseudo3D observation files: %s" % summary.get("files"))
    print("pseudo3D used rate: %s" % summary.get("pseudo3d_used_rate"))
    return summary


def _load_yaml(path: Path) -> Dict[str, Any]:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def _scenes_for_subsets(subsets_cfg: Dict[str, Any], subset_names: Optional[List[str]]) -> Optional[List[str]]:
    if subset_names is None:
        return None
    scenes = []
    for subset in subset_names:
        data = subsets_cfg.get(subset, {})
        if isinstance(data, dict):
            scenes.extend([str(item) for item in data.get("scenes", [])])
    return scenes or None


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build baseline_v2 pseudo-3D Observation3D files.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--subsets", nargs="+", default=None)
    parser.add_argument("--scenes", nargs="+", default=None)
    parser.add_argument("--camera-ids", nargs="+", default=None)
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

