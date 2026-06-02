"""Run local tracking over many Observation3D JSONL files."""

import argparse
import csv
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import yaml

from deep_oc_sort_3d.observations.observation_io import read_observations_jsonl
from deep_oc_sort_3d.tracking.local_tracker import LocalObservationTracker
from deep_oc_sort_3d.tracking.track_io import write_local_tracks_csv


SUMMARY_FIELDS = [
    "subset",
    "scene_name",
    "camera_id",
    "observations_path",
    "tracks_path",
    "num_observations",
    "num_track_records",
    "num_active_tracks",
    "status",
    "error_message",
]


def run_batch_local_tracking(args: Any) -> None:
    """Run local tracking file-by-file over a pipeline observations directory."""
    config = _load_tracking_config(args.config)
    options = _resolve_options(config, args)
    if options["run_root"] is None:
        raise ValueError("Provide --run-root or set tracking.run_root in the config.")
    files = _find_observation_files(
        run_root=Path(options["run_root"]),
        subsets=options.get("subsets"),
        scenes=options.get("scenes"),
        camera_ids=options.get("camera_ids"),
    )
    rows = []
    for path in _progress_iter(files, options["progress"], "batch local tracking files"):
        rows.append(_run_one_file(path, options))
    summary_path = Path(options["output_root"]) / "summaries" / "local_tracking_summary.csv"
    _write_summary(rows, summary_path)
    print("files: %d" % len(files))
    print("errors: %d" % len([row for row in rows if row["status"] == "error"]))
    print("Wrote %s" % summary_path)


def _run_one_file(path: Path, options: Dict[str, Any]) -> Dict[str, Any]:
    subset, scene_name, camera_id = _parse_observation_path(path)
    tracks_path = Path(options["output_root"]) / subset / scene_name / ("%s.csv" % camera_id)
    if tracks_path.exists() and not options["overwrite"]:
        return _summary_row(subset, scene_name, camera_id, path, tracks_path, 0, 0, 0, "skipped_existing", "")
    try:
        observations = read_observations_jsonl(path)
        tracker = LocalObservationTracker(
            mode=options["mode"],
            min_confidence=options["min_confidence"],
            min_hits=options["min_hits"],
            max_misses=options["max_misses"],
            cost_threshold=options["cost_threshold"],
            max_3d_distance=options["max_3d_distance"],
            min_iou=options["min_iou"],
            class_must_match=options["class_must_match"],
            max_detections_per_frame=options.get("max_detections_per_frame"),
        )
        desc = "%s %s %s" % (subset, scene_name, camera_id)
        records = tracker.run(observations, show_progress=options["progress"])
        write_local_tracks_csv(records, tracks_path)
        summary = tracker.summary()
        return _summary_row(
            subset,
            scene_name,
            camera_id,
            path,
            tracks_path,
            len(observations),
            len(records),
            int(summary.get("num_active_tracks", 0)),
            "ok",
            "",
        )
    except Exception as exc:
        return _summary_row(subset, scene_name, camera_id, path, tracks_path, 0, 0, 0, "error", str(exc))


def _find_observation_files(
    run_root: Path,
    subsets: Optional[List[str]],
    scenes: Optional[List[str]],
    camera_ids: Optional[List[str]],
) -> List[Path]:
    base = run_root / "observations3d"
    files = []
    subset_set = None if subsets is None else set(subsets)
    scene_set = None if scenes is None else set(scenes)
    camera_set = None if camera_ids is None else set(camera_ids)
    for path in sorted(base.rglob("*.jsonl")):
        subset, scene_name, camera_id = _parse_observation_path(path)
        if subset_set is not None and subset not in subset_set:
            continue
        if scene_set is not None and scene_name not in scene_set:
            continue
        if camera_set is not None and camera_id not in camera_set:
            continue
        files.append(path)
    return files


def _parse_observation_path(path: Path) -> Tuple[str, str, str]:
    parts = list(Path(path).parts)
    index = parts.index("observations3d")
    subset = parts[index + 1]
    scene_name = parts[index + 2]
    camera_id = Path(parts[index + 3]).stem
    return subset, scene_name, camera_id


def _load_tracking_config(path: Optional[Path]) -> Dict[str, Any]:
    if path is None:
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        return {}
    tracking = data.get("tracking", {})
    if isinstance(tracking, dict):
        return tracking
    return {}


def _resolve_options(config: Dict[str, Any], args: Any) -> Dict[str, Any]:
    return {
        "run_root": _value(args.run_root, config.get("run_root"), None),
        "output_root": _value(args.output_root, config.get("output_root"), "output/local_tracks/debug"),
        "subsets": _list_value(args.subsets, config.get("subsets")),
        "scenes": _list_value(args.scenes, config.get("scenes")),
        "camera_ids": _list_value(args.camera_ids, config.get("camera_ids")),
        "mode": _value(args.mode, config.get("mode"), "hybrid"),
        "min_confidence": float(_value(args.min_confidence, config.get("min_confidence"), 0.01)),
        "min_hits": int(_value(args.min_hits, config.get("min_hits"), 2)),
        "max_misses": int(_value(args.max_misses, config.get("max_misses"), 30)),
        "cost_threshold": float(_value(args.cost_threshold, config.get("cost_threshold"), 0.7)),
        "max_3d_distance": float(_value(args.max_3d_distance, config.get("max_3d_distance"), 3.0)),
        "min_iou": float(_value(args.min_iou, config.get("min_iou"), 0.05)),
        "class_must_match": bool(_value(args.class_must_match, config.get("class_must_match"), True)),
        "max_detections_per_frame": _optional_int(_value(args.max_detections_per_frame, config.get("max_detections_per_frame"), None)),
        "progress": bool(_value(args.progress, config.get("progress"), True)),
        "overwrite": bool(args.overwrite),
    }


def _summary_row(
    subset: str,
    scene_name: str,
    camera_id: str,
    observations_path: Path,
    tracks_path: Path,
    num_observations: int,
    num_track_records: int,
    num_active_tracks: int,
    status: str,
    error_message: str,
) -> Dict[str, Any]:
    return {
        "subset": subset,
        "scene_name": scene_name,
        "camera_id": camera_id,
        "observations_path": str(observations_path),
        "tracks_path": str(tracks_path),
        "num_observations": int(num_observations),
        "num_track_records": int(num_track_records),
        "num_active_tracks": int(num_active_tracks),
        "status": status,
        "error_message": error_message,
    }


def _write_summary(rows: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


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


def _value(cli_value: Any, config_value: Any, default: Any) -> Any:
    if cli_value is not None:
        return cli_value
    if config_value is not None:
        return config_value
    return default


def _list_value(cli_value: Any, config_value: Any) -> Any:
    value = _value(cli_value, config_value, None)
    if value is None:
        return None
    if isinstance(value, str):
        return [value]
    return [str(item) for item in list(value)]


def _optional_int(value: Any) -> Any:
    if value is None:
        return None
    return int(value)


def build_arg_parser() -> argparse.ArgumentParser:
    """Build CLI parser."""
    parser = argparse.ArgumentParser(description="Run batch local tracking.")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--run-root", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--subsets", nargs="+", default=None)
    parser.add_argument("--scenes", nargs="+", default=None)
    parser.add_argument("--camera-ids", nargs="+", default=None)
    parser.add_argument("--mode", choices=["oracle_3d", "hybrid", "bbox2d"], default=None)
    parser.add_argument("--min-confidence", type=float, default=None)
    parser.add_argument("--min-hits", type=int, default=None)
    parser.add_argument("--max-misses", type=int, default=None)
    parser.add_argument("--cost-threshold", type=float, default=None)
    parser.add_argument("--max-3d-distance", type=float, default=None)
    parser.add_argument("--min-iou", type=float, default=None)
    parser.add_argument("--max-detections-per-frame", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    class_group = parser.add_mutually_exclusive_group()
    class_group.add_argument("--class-must-match", dest="class_must_match", action="store_true", default=None)
    class_group.add_argument("--no-class-must-match", dest="class_must_match", action="store_false")
    progress_group = parser.add_mutually_exclusive_group()
    progress_group.add_argument("--progress", dest="progress", action="store_true", default=None)
    progress_group.add_argument("--no-progress", dest="progress", action="store_false")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    run_batch_local_tracking(args)


if __name__ == "__main__":
    main()
