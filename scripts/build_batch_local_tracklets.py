"""Build local tracklets for a directory of local track CSV files."""

import argparse
import csv
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import yaml

from deep_oc_sort_3d.scripts.build_local_tracklets import _build_with_progress
from deep_oc_sort_3d.tracking.track_io import read_local_tracks_csv
from deep_oc_sort_3d.tracklets.tracklet_builder import LocalTrackletBuilder
from deep_oc_sort_3d.tracklets.tracklet_io import write_tracklets_csv, write_tracklets_jsonl
from deep_oc_sort_3d.tracklets.tracklet_summary import (
    summarize_tracklet_files,
    write_tracklet_summary_csv,
    write_tracklet_summary_json,
)


def build_batch_local_tracklets(args: Any) -> None:
    """Build tracklets file-by-file."""
    config = _load_config(args.config)
    options = _resolve_options(config, args)
    if options["tracking_root"] is None:
        raise ValueError("Provide --tracking-root or set tracklets.tracking_root in the config.")
    files = _find_track_csv_files(
        Path(options["tracking_root"]),
        options.get("subsets"),
        options.get("scenes"),
        options.get("camera_ids"),
    )
    output_jsonls = []
    rows = []
    for path in _progress_iter(files, options["progress"], "batch local tracklets"):
        row, jsonl_path = _process_one_file(path, options)
        rows.append(row)
        if jsonl_path is not None:
            output_jsonls.append(jsonl_path)
    summary = summarize_tracklet_files(output_jsonls)
    summary["build_rows"] = rows
    summary_root = Path(options["output_root"]) / "summaries"
    write_tracklet_summary_csv(summary, summary_root / "tracklet_summary.csv")
    write_tracklet_summary_json(summary, summary_root / "tracklet_summary.json")
    print("files: %d" % len(files))
    print("errors: %d" % len([row for row in rows if row.get("status") == "error"]))
    print("Wrote summary: %s" % (summary_root / "tracklet_summary.json"))


def _process_one_file(path: Path, options: Dict[str, Any]) -> Tuple[Dict[str, Any], Optional[Path]]:
    subset, scene_name, camera_id = _parse_track_path(Path(options["tracking_root"]), path)
    output_dir = Path(options["output_root"]) / subset / scene_name
    output_csv = output_dir / ("%s_tracklets.csv" % camera_id)
    output_jsonl = output_dir / ("%s_tracklets.jsonl" % camera_id)
    if output_csv.exists() and output_jsonl.exists() and not options["overwrite"]:
        return _row(path, output_csv, output_jsonl, 0, 0, "skipped_existing", ""), output_jsonl
    try:
        records = read_local_tracks_csv(path)
        builder = LocalTrackletBuilder(
            min_length=options["min_length"],
            min_mean_confidence=options["min_mean_confidence"],
            smooth_trajectory=options["smooth_trajectory"],
            smoothing_window=options["smoothing_window"],
        )
        desc = "%s %s %s" % (subset, scene_name, camera_id)
        tracklets = _build_with_progress(builder, records, options["progress"], desc)
        write_tracklets_csv(tracklets, output_csv)
        write_tracklets_jsonl(tracklets, output_jsonl)
        return _row(path, output_csv, output_jsonl, len(records), len(tracklets), "ok", ""), output_jsonl
    except Exception as exc:
        return _row(path, output_csv, output_jsonl, 0, 0, "error", str(exc)), None


def _find_track_csv_files(
    tracking_root: Path,
    subsets: Optional[List[str]],
    scenes: Optional[List[str]],
    camera_ids: Optional[List[str]],
) -> List[Path]:
    subset_set = None if subsets is None else set(subsets)
    scene_set = None if scenes is None else set(scenes)
    camera_set = None if camera_ids is None else set(camera_ids)
    files = []
    for path in sorted(tracking_root.rglob("*.csv")):
        if not _looks_like_track_csv(path):
            continue
        subset, scene_name, camera_id = _parse_track_path(tracking_root, path)
        if subset_set is not None and subset not in subset_set:
            continue
        if scene_set is not None and scene_name not in scene_set:
            continue
        if camera_set is not None and camera_id not in camera_set:
            continue
        files.append(path)
    return files


def _looks_like_track_csv(path: Path) -> bool:
    required = set(["scene_id", "frame_id", "local_track_id", "detection_id", "class_id"])
    try:
        with path.open("r", newline="", encoding="utf-8") as handle:
            header = next(csv.reader(handle), [])
    except (IOError, OSError):
        return False
    return required.issubset(set(header))


def _parse_track_path(root: Path, path: Path) -> Tuple[str, str, str]:
    rel = path.relative_to(root)
    parts = list(rel.parts)
    if len(parts) < 3:
        return "unknown", "unknown", path.stem
    return parts[0], parts[1], Path(parts[2]).stem


def _row(
    input_path: Path,
    output_csv: Path,
    output_jsonl: Path,
    num_records: int,
    num_tracklets: int,
    status: str,
    error_message: str,
) -> Dict[str, Any]:
    return {
        "input_path": str(input_path),
        "output_csv": str(output_csv),
        "output_jsonl": str(output_jsonl),
        "num_records": int(num_records),
        "num_tracklets": int(num_tracklets),
        "status": status,
        "error_message": error_message,
    }


def _load_config(path: Optional[Path]) -> Dict[str, Any]:
    if path is None:
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        return {}
    tracklets = data.get("tracklets", {})
    if isinstance(tracklets, dict):
        return tracklets
    return {}


def _resolve_options(config: Dict[str, Any], args: Any) -> Dict[str, Any]:
    return {
        "tracking_root": _value(args.tracking_root, config.get("tracking_root"), None),
        "output_root": _value(args.output_root, config.get("output_root"), "output/tracklets/debug"),
        "subsets": _list_value(args.subsets, config.get("subsets")),
        "scenes": _list_value(args.scenes, config.get("scenes")),
        "camera_ids": _list_value(args.camera_ids, config.get("camera_ids")),
        "min_length": int(_value(args.min_length, config.get("min_length"), 3)),
        "min_mean_confidence": float(_value(args.min_mean_confidence, config.get("min_mean_confidence"), 0.01)),
        "smooth_trajectory": bool(_value(args.smooth_trajectory, config.get("smooth_trajectory"), True)),
        "smoothing_window": int(_value(args.smoothing_window, config.get("smoothing_window"), 5)),
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
    parser = argparse.ArgumentParser(description="Build local tracklets in batch.")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--tracking-root", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--subsets", nargs="+", default=None)
    parser.add_argument("--scenes", nargs="+", default=None)
    parser.add_argument("--camera-ids", nargs="+", default=None)
    parser.add_argument("--min-length", type=int, default=None)
    parser.add_argument("--min-mean-confidence", type=float, default=None)
    parser.add_argument("--smoothing-window", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    smooth_group = parser.add_mutually_exclusive_group()
    smooth_group.add_argument("--smooth-trajectory", dest="smooth_trajectory", action="store_true", default=None)
    smooth_group.add_argument("--no-smooth-trajectory", dest="smooth_trajectory", action="store_false")
    progress_group = parser.add_mutually_exclusive_group()
    progress_group.add_argument("--progress", dest="progress", action="store_true", default=None)
    progress_group.add_argument("--no-progress", dest="progress", action="store_false")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    build_batch_local_tracklets(args)


if __name__ == "__main__":
    main()
