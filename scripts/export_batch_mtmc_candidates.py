"""Batch export MTMC candidates from local tracklet files."""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import yaml

from deep_oc_sort_3d.mtmc.candidate_builder import MTMCCandidateBuilder
from deep_oc_sort_3d.mtmc.candidate_io import write_candidates_csv, write_candidates_jsonl
from deep_oc_sort_3d.mtmc.candidate_summary import (
    summarize_candidates,
    write_candidate_summary_csv,
    write_candidate_summary_json,
)
from deep_oc_sort_3d.scripts.export_mtmc_candidates import _build_with_progress
from deep_oc_sort_3d.tracklets.tracklet_io import read_tracklets_file


def export_batch_mtmc_candidates(args: Any) -> None:
    """Export MTMC candidates file-by-file."""
    config = _load_config(args.config)
    options = _resolve_options(config, args)
    if options["tracklet_root"] is None:
        raise ValueError("Provide --tracklet-root or set candidates.tracklet_root in config.")
    files = _find_tracklet_files(
        Path(options["tracklet_root"]),
        options.get("subsets"),
        options.get("scenes"),
        options.get("camera_ids"),
    )
    rows = []
    all_candidates = []
    for path in _progress_iter(files, options["progress"], "batch MTMC candidates"):
        row, candidates = _process_one_file(path, options)
        rows.append(row)
        all_candidates.extend(candidates)
    summary = summarize_candidates(all_candidates)
    summary["files"] = rows
    summary_root = Path(options["output_root"]) / "summaries"
    write_candidate_summary_json(summary, summary_root / "candidate_summary.json")
    write_candidate_summary_csv(summary, summary_root / "candidate_summary.csv")
    print("files: %d" % len(files))
    print("errors: %d" % len([row for row in rows if row.get("status") == "error"]))
    print("kept candidates: %s" % summary.get("kept_candidates"))
    print("Wrote summary: %s" % (summary_root / "candidate_summary.json"))


def _process_one_file(path: Path, options: Dict[str, Any]) -> Tuple[Dict[str, Any], List[Any]]:
    subset, scene_name, camera_id = _parse_tracklet_path(Path(options["tracklet_root"]), path)
    output_dir = Path(options["output_root"]) / subset / scene_name
    output_csv = output_dir / ("%s_candidates.csv" % camera_id)
    output_jsonl = output_dir / ("%s_candidates.jsonl" % camera_id)
    if output_csv.exists() and output_jsonl.exists() and not options["overwrite"]:
        return _row(path, output_csv, output_jsonl, 0, 0, 0, "skipped_existing", ""), []
    try:
        tracklets = read_tracklets_file(path)
        builder = MTMCCandidateBuilder(
            min_length=options["min_length"],
            min_mean_confidence=options["min_mean_confidence"],
            allowed_quality_flags=options["allowed_quality_flags"],
            require_valid_for_mtmc=options["require_valid_for_mtmc"],
            require_3d=options["require_3d"],
            trajectory_sample_rate=options["trajectory_sample_rate"],
            max_trajectory_points=options["max_trajectory_points"],
            class_allowlist=options["class_allowlist"],
            class_blocklist=options["class_blocklist"],
        )
        desc = "%s %s %s" % (subset, scene_name, camera_id)
        candidates = _build_with_progress(builder, tracklets, subset, options["progress"], desc)
        exported = candidates if options["export_rejected"] else [item for item in candidates if item.is_candidate]
        write_candidates_csv(exported, output_csv)
        write_candidates_jsonl(exported, output_jsonl)
        return (
            _row(path, output_csv, output_jsonl, len(tracklets), len(candidates), len(exported), "ok", ""),
            candidates,
        )
    except Exception as exc:
        return _row(path, output_csv, output_jsonl, 0, 0, 0, "error", str(exc)), []


def _find_tracklet_files(
    root: Path,
    subsets: Optional[List[str]],
    scenes: Optional[List[str]],
    camera_ids: Optional[List[str]],
) -> List[Path]:
    jsonl_files = sorted(root.rglob("*_tracklets.jsonl"))
    files = jsonl_files if jsonl_files else sorted(root.rglob("*_tracklets.csv"))
    subset_set = None if subsets is None else set(subsets)
    scene_set = None if scenes is None else set(scenes)
    camera_set = None if camera_ids is None else set(camera_ids)
    output = []
    for path in files:
        subset, scene_name, camera_id = _parse_tracklet_path(root, path)
        if subset_set is not None and subset not in subset_set:
            continue
        if scene_set is not None and scene_name not in scene_set:
            continue
        if camera_set is not None and camera_id not in camera_set:
            continue
        output.append(path)
    return output


def _parse_tracklet_path(root: Path, path: Path) -> Tuple[str, str, str]:
    rel = path.relative_to(root)
    parts = list(rel.parts)
    if len(parts) < 3:
        return "unknown", "unknown", path.stem.replace("_tracklets", "")
    camera = Path(parts[2]).stem.replace("_tracklets", "")
    return parts[0], parts[1], camera


def _row(
    input_path: Path,
    output_csv: Path,
    output_jsonl: Path,
    num_tracklets: int,
    num_candidates_including_rejected: int,
    num_exported: int,
    status: str,
    error_message: str,
) -> Dict[str, Any]:
    return {
        "input_path": str(input_path),
        "output_csv": str(output_csv),
        "output_jsonl": str(output_jsonl),
        "num_tracklets": int(num_tracklets),
        "num_candidates_including_rejected": int(num_candidates_including_rejected),
        "num_exported": int(num_exported),
        "status": status,
        "error_message": error_message,
    }


def _load_config(path: Optional[Path]) -> Dict[str, Any]:
    if path is None:
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        return {}
    candidates = data.get("candidates", {})
    if isinstance(candidates, dict):
        return candidates
    return {}


def _resolve_options(config: Dict[str, Any], args: Any) -> Dict[str, Any]:
    return {
        "tracklet_root": _value(args.tracklet_root, config.get("tracklet_root"), None),
        "output_root": _value(args.output_root, config.get("output_root"), "output/mtmc_candidates/debug"),
        "subsets": _list_value(args.subsets, config.get("subsets")),
        "scenes": _list_value(args.scenes, config.get("scenes")),
        "camera_ids": _list_value(args.camera_ids, config.get("camera_ids")),
        "min_length": int(_value(args.min_length, config.get("min_length"), 3)),
        "min_mean_confidence": float(_value(args.min_mean_confidence, config.get("min_mean_confidence"), 0.01)),
        "allowed_quality_flags": _list_value(args.allowed_quality_flags, config.get("allowed_quality_flags")),
        "require_valid_for_mtmc": bool(_value(args.require_valid_for_mtmc, config.get("require_valid_for_mtmc"), True)),
        "require_3d": bool(_value(args.require_3d, config.get("require_3d"), False)),
        "trajectory_sample_rate": int(_value(args.trajectory_sample_rate, config.get("trajectory_sample_rate"), 5)),
        "max_trajectory_points": int(_value(args.max_trajectory_points, config.get("max_trajectory_points"), 50)),
        "class_allowlist": _list_value(args.class_allowlist, config.get("class_allowlist")),
        "class_blocklist": _list_value(args.class_blocklist, config.get("class_blocklist")),
        "export_rejected": bool(_value(args.export_rejected, config.get("export_rejected"), False)),
        "progress": bool(_value(args.progress, config.get("progress"), True)),
        "overwrite": bool(args.overwrite),
    }


def _value(cli_value: Any, config_value: Any, default: Any) -> Any:
    if cli_value is not None:
        return cli_value
    if config_value is not None:
        return config_value
    return default


def _list_value(cli_value: Any, config_value: Any) -> Optional[List[str]]:
    value = _value(cli_value, config_value, None)
    if value is None:
        return None
    if isinstance(value, str):
        return [value]
    return [str(item) for item in list(value)]


def _progress_iter(values: List[Path], show_progress: bool, desc: str) -> Iterable[Path]:
    if not show_progress:
        return values
    try:
        from tqdm import tqdm
    except ImportError:
        return _print_progress_iter(values, desc)
    return tqdm(values, desc=desc, unit="file")


def _print_progress_iter(values: List[Path], desc: str) -> Iterable[Path]:
    total = len(values)
    for index, value in enumerate(values):
        print("%s: file %d/%d %s" % (desc, index + 1, total, value))
        yield value


def build_arg_parser() -> argparse.ArgumentParser:
    """Build CLI parser."""
    parser = argparse.ArgumentParser(description="Batch export MTMC candidates.")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--tracklet-root", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--subsets", nargs="+", default=None)
    parser.add_argument("--scenes", nargs="+", default=None)
    parser.add_argument("--camera-ids", nargs="+", default=None)
    parser.add_argument("--min-length", type=int, default=None)
    parser.add_argument("--min-mean-confidence", type=float, default=None)
    parser.add_argument("--allowed-quality-flags", nargs="+", default=None)
    parser.add_argument("--trajectory-sample-rate", type=int, default=None)
    parser.add_argument("--max-trajectory-points", type=int, default=None)
    parser.add_argument("--class-allowlist", nargs="+", default=None)
    parser.add_argument("--class-blocklist", nargs="+", default=None)
    parser.add_argument("--export-rejected", action="store_true", default=None)
    parser.add_argument("--overwrite", action="store_true")
    valid_group = parser.add_mutually_exclusive_group()
    valid_group.add_argument("--require-valid-for-mtmc", dest="require_valid_for_mtmc", action="store_true", default=None)
    valid_group.add_argument("--no-require-valid-for-mtmc", dest="require_valid_for_mtmc", action="store_false")
    d_group = parser.add_mutually_exclusive_group()
    d_group.add_argument("--require-3d", dest="require_3d", action="store_true", default=None)
    d_group.add_argument("--no-require-3d", dest="require_3d", action="store_false")
    progress_group = parser.add_mutually_exclusive_group()
    progress_group.add_argument("--progress", dest="progress", action="store_true", default=None)
    progress_group.add_argument("--no-progress", dest="progress", action="store_false")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    export_batch_mtmc_candidates(args)


if __name__ == "__main__":
    main()
