"""Run local single-camera tracking on one Observation3D JSONL file."""

import argparse
import json
from pathlib import Path
from typing import Any, Dict

from deep_oc_sort_3d.observations.observation_io import read_observations_jsonl
from deep_oc_sort_3d.tracking.local_tracker import LocalObservationTracker
from deep_oc_sort_3d.tracking.track_io import write_local_tracks_csv, write_local_tracks_jsonl


def run_local_tracking(args: Any) -> None:
    """Run local tracking over one observations file."""
    observations = read_observations_jsonl(args.observations)
    tracker = LocalObservationTracker(
        mode=args.mode,
        min_confidence=args.min_confidence,
        min_hits=args.min_hits,
        max_misses=args.max_misses,
        cost_threshold=args.cost_threshold,
        max_3d_distance=args.max_3d_distance,
        min_iou=args.min_iou,
        class_must_match=args.class_must_match,
        max_detections_per_frame=args.max_detections_per_frame,
        per_class_conf_thresholds=_load_per_class_conf(args.per_class_conf_json),
    )
    records = tracker.run(observations, show_progress=args.progress)
    write_local_tracks_csv(records, args.output)
    if args.jsonl_output is not None:
        write_local_tracks_jsonl(records, args.jsonl_output)
    summary = tracker.summary()
    summary["num_track_records"] = len(records)
    if args.summary_output is not None:
        args.summary_output.parent.mkdir(parents=True, exist_ok=True)
        args.summary_output.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print("Wrote tracks: %s" % args.output)
    print("track_records: %d" % len(records))
    print("active_tracks: %s" % summary.get("num_active_tracks"))


def _load_per_class_conf(value: Any) -> Dict[str, float]:
    if value is None:
        return {}
    path = Path(str(value))
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
    else:
        data = json.loads(str(value))
    return {str(key): float(item) for key, item in data.items()}


def build_arg_parser() -> argparse.ArgumentParser:
    """Build CLI parser."""
    parser = argparse.ArgumentParser(description="Run local single-camera tracking.")
    parser.add_argument("--observations", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--jsonl-output", type=Path, default=None)
    parser.add_argument("--summary-output", type=Path, default=None)
    parser.add_argument("--mode", choices=["oracle_3d", "hybrid", "bbox2d"], default="hybrid")
    parser.add_argument("--min-confidence", type=float, default=0.01)
    parser.add_argument("--min-hits", type=int, default=2)
    parser.add_argument("--max-misses", type=int, default=30)
    parser.add_argument("--cost-threshold", type=float, default=0.7)
    parser.add_argument("--max-3d-distance", type=float, default=3.0)
    parser.add_argument("--min-iou", type=float, default=0.05)
    parser.add_argument("--max-detections-per-frame", type=int, default=None)
    parser.add_argument("--per-class-conf-json", default=None)
    class_group = parser.add_mutually_exclusive_group()
    class_group.add_argument("--class-must-match", dest="class_must_match", action="store_true")
    class_group.add_argument("--no-class-must-match", dest="class_must_match", action="store_false")
    progress_group = parser.add_mutually_exclusive_group()
    progress_group.add_argument("--progress", dest="progress", action="store_true")
    progress_group.add_argument("--no-progress", dest="progress", action="store_false")
    parser.set_defaults(class_must_match=True, progress=True)
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    run_local_tracking(args)


if __name__ == "__main__":
    main()
