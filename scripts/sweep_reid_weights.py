"""Run a small ReID appearance-weight sweep."""

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import yaml

from deep_oc_sort_3d.scripts.compare_reid_global_runs import summarize_global_run
from deep_oc_sort_3d.scripts.run_batch_global_mtmc_with_reid import run_batch_global_mtmc_with_reid


def sweep_reid_weights(args: Any) -> None:
    """Run batch ReID association for multiple appearance weights."""
    config = _load_sweep_config(args.sweep_config)
    candidates_root = args.candidates_root or _path_from_config(config, "candidates_root")
    reid_root = args.reid_root or _path_from_config(config, "reid_root")
    output_root = args.output_root or _path_from_config(config, "output_root")
    base_config = args.base_config or _path_from_config(config, "base_config")
    weights = args.weights if args.weights else [float(item) for item in config.get("weights", [0.05, 0.10, 0.20])]
    subsets = args.subsets if args.subsets is not None else config.get("subsets", None)
    if candidates_root is None or reid_root is None or output_root is None or base_config is None:
        raise ValueError("candidates_root, reid_root, output_root, and base_config are required")
    run_roots = []
    names = []
    for weight in _progress_iter(weights, args.progress, "ReID weight sweep", "weight"):
        name = "w%03d" % int(round(float(weight) * 1000.0))
        run_root = output_root / name
        batch_args = _BatchArgs(
            candidates_root=candidates_root,
            reid_root=reid_root,
            output_root=run_root,
            subsets=subsets,
            scenes=args.scenes,
            config=base_config,
            appearance_weight=float(weight),
            use_reid=True,
            class_names=args.class_names,
            max_candidates_per_scene=args.max_candidates_per_scene,
            overwrite=args.overwrite,
            progress=args.progress,
        )
        run_batch_global_mtmc_with_reid(batch_args)
        run_roots.append(run_root)
        names.append(name)
    _write_sweep_comparison(output_root, names, run_roots)


class _BatchArgs:
    """Tiny namespace for forwarding batch sweep arguments."""

    def __init__(
        self,
        candidates_root: Path,
        reid_root: Path,
        output_root: Path,
        subsets: Optional[List[str]],
        scenes: Optional[List[str]],
        config: Path,
        appearance_weight: float,
        use_reid: bool,
        class_names: Optional[List[str]],
        max_candidates_per_scene: Optional[int],
        overwrite: bool,
        progress: bool,
    ) -> None:
        self.candidates_root = candidates_root
        self.reid_root = reid_root
        self.output_root = output_root
        self.subsets = subsets
        self.scenes = scenes
        self.config = config
        self.appearance_weight = appearance_weight
        self.use_reid = use_reid
        self.class_names = class_names
        self.max_candidates_per_scene = max_candidates_per_scene
        self.overwrite = overwrite
        self.progress = progress


def _write_sweep_comparison(output_root: Path, names: List[str], run_roots: List[Path]) -> None:
    rows = [summarize_global_run(name, root) for name, root in zip(names, run_roots)]
    summary_root = output_root / "summaries"
    summary_root.mkdir(parents=True, exist_ok=True)
    fields = sorted(set([key for row in rows for key in row.keys()])) if rows else ["name"]
    csv_path = summary_root / "reid_ablation_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    (summary_root / "reid_ablation_summary.json").write_text(
        json.dumps({"runs": rows}, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print("Wrote %s" % csv_path)


def _load_sweep_config(path: Optional[Path]) -> Dict[str, Any]:
    if path is None:
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        return {}
    section = data.get("sweep", data)
    return section if isinstance(section, dict) else {}


def _path_from_config(config: Dict[str, Any], key: str) -> Optional[Path]:
    value = config.get(key)
    if value in (None, ""):
        return None
    return Path(str(value))


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
        print("%s: %d/%d %s" % (desc, index + 1, total, str(value)))
        yield value


def build_arg_parser() -> argparse.ArgumentParser:
    """Build CLI parser."""
    parser = argparse.ArgumentParser(description="Sweep ReID appearance weights for global MTMC association.")
    parser.add_argument("--sweep-config", type=Path, default=None)
    parser.add_argument("--candidates-root", type=Path, default=None)
    parser.add_argument("--reid-root", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--weights", nargs="+", type=float, default=None)
    parser.add_argument("--subsets", nargs="+", default=None)
    parser.add_argument("--scenes", nargs="+", default=None)
    parser.add_argument("--base-config", type=Path, default=None)
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
    sweep_reid_weights(args)


if __name__ == "__main__":
    main()
