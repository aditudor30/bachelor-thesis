"""Generate standard V2 LocalTrackRecord files using ByteTrack-style IDs."""

import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from deep_oc_sort_3d.bytetrack_local_pipeline.bytetrack_io import progress_iter, write_csv, write_json
from deep_oc_sort_3d.bytetrack_local_pipeline.bytetrack_pipeline_config import configured_subsets
from deep_oc_sort_3d.local_tracker_benchmark.bytetrack_style_tracker import ByteTrackStyleTracker
from deep_oc_sort_3d.local_tracker_benchmark.detection_loader import group_detections_by_frame, load_camera_detections
from deep_oc_sort_3d.observations.observation_io import read_observations_jsonl
from deep_oc_sort_3d.tracking.association import bbox_iou_xyxy
from deep_oc_sort_3d.tracking.track_io import write_local_tracks_csv
from deep_oc_sort_3d.tracking.track_types import LocalTrackRecord


def run_bytetrack_local_tracking(
    config: Dict[str, Any],
    progress: bool = True,
    overwrite: bool = False,
) -> Dict[str, Any]:
    """Run per-camera ByteTrack and preserve V2 observation geometry."""
    paths = config.get("paths", {})
    output_root = Path(str(paths.get("output_local_tracks_root")))
    if overwrite and output_root.exists():
        shutil.rmtree(str(output_root))
    inventory = build_full_inventory(config)
    summaries = []
    for item in progress_iter(inventory, progress, "ByteTrack local cameras"):
        try:
            summary = run_one_camera(config, item)
        except Exception as exc:
            summary = dict(item)
            summary.update({"status": "error", "error": str(exc), "num_track_records": 0})
        summaries.append(summary)
    summary_root = output_root / "summaries"
    write_csv(summary_root / "local_tracking_summary.csv", summaries)
    aggregate = {
        "files": len(summaries),
        "ok": sum(1 for row in summaries if row.get("status") == "ok"),
        "errors": sum(1 for row in summaries if row.get("status") == "error"),
        "num_track_records": sum(int(row.get("num_track_records", 0)) for row in summaries),
        "missing_observation_records": sum(int(row.get("missing_observation_records", 0)) for row in summaries),
    }
    write_json(summary_root / "local_tracking_summary.json", aggregate)
    return aggregate


def build_full_inventory(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Inventory detection and V2 observation pairs for configured scenes."""
    paths = config.get("paths", {})
    detection_root = Path(str(paths.get("yolo_pipeline_root"))) / "detections2d"
    observations_root = Path(str(paths.get("v2_observations_root")))
    output = []
    for subset, split, scene_name in configured_subsets(config, "full_rerun"):
        scene_root = detection_root / subset / scene_name
        for detection_path in sorted(scene_root.glob("*.csv")) if scene_root.exists() else []:
            output.append(
                {
                    "subset": subset,
                    "split": split,
                    "scene_name": scene_name,
                    "camera_id": detection_path.stem,
                    "detections_path": str(detection_path),
                    "observations_path": str(observations_root / subset / scene_name / (detection_path.stem + ".jsonl")),
                }
            )
    return output


def run_one_camera(config: Dict[str, Any], item: Dict[str, Any]) -> Dict[str, Any]:
    """Track one camera and attach each result to its V2 Observation3D."""
    settings = config.get("bytetrack_style", {})
    detections = load_camera_detections(item, float(settings.get("min_confidence_for_input", 0.001)))
    observations = read_observations_jsonl(Path(str(item.get("observations_path"))))
    exact, by_frame = _observation_lookups(observations)
    tracker = ByteTrackStyleTracker(settings)
    benchmark_records = tracker.run(group_detections_by_frame(detections))
    records = []
    missing = 0
    for tracked in benchmark_records:
        observation = exact.get((tracked.frame_id, tracked.detection_id))
        if observation is None:
            observation = _fallback_observation(tracked, by_frame.get(tracked.frame_id, []))
        if observation is None:
            missing += 1
            continue
        records.append(local_record_from_benchmark(tracked, observation))
    output_path = (
        Path(str(config.get("paths", {}).get("output_local_tracks_root")))
        / str(item.get("subset"))
        / str(item.get("scene_name"))
        / (str(item.get("camera_id")) + ".csv")
    )
    write_local_tracks_csv(records, output_path)
    return {
        "subset": item.get("subset"),
        "scene_name": item.get("scene_name"),
        "camera_id": item.get("camera_id"),
        "status": "ok",
        "num_input_detections": len(detections),
        "num_observations": len(observations),
        "num_track_records": len(records),
        "num_active_tracks": len(set(record.local_track_id for record in records)),
        "missing_observation_records": missing,
        "output_path": str(output_path),
    }


def local_record_from_benchmark(tracked: Any, observation: Any) -> LocalTrackRecord:
    """Convert one ByteTrack record and its observation to standard pipeline format."""
    return LocalTrackRecord(
        scene_id=int(observation.scene_id),
        scene_name=str(observation.scene_name),
        split=str(observation.split),
        camera_id=str(observation.camera_id),
        frame_id=int(observation.frame_id),
        local_track_id=int(tracked.track_id),
        detection_id=int(observation.detection_id),
        class_id=int(observation.class_id),
        class_name=str(observation.class_name),
        confidence=float(observation.confidence),
        bbox_xyxy=tuple(float(value) for value in observation.bbox_xyxy),
        bbox_xywh=tuple(float(value) for value in observation.bbox_xywh),
        center_3d=None if observation.center_3d is None else observation.center_3d.copy(),
        dimensions_3d=None if observation.dimensions_3d is None else observation.dimensions_3d.copy(),
        yaw=None if observation.yaw is None else float(observation.yaw),
        matched_gt_object_id=None if observation.object_id is None else int(observation.object_id),
        matched_gt=bool(observation.matched_gt),
        track_age=int(tracked.track_age),
        track_hits=int(tracked.track_hits),
        track_misses=int(tracked.track_misses),
        track_state=str(tracked.track_state),
    )


def _observation_lookups(observations: List[Any]) -> Tuple[Dict[Tuple[int, int], Any], Dict[int, List[Any]]]:
    exact = {}
    by_frame = {}
    for observation in observations:
        exact[(int(observation.frame_id), int(observation.detection_id))] = observation
        by_frame.setdefault(int(observation.frame_id), []).append(observation)
    return exact, by_frame


def _fallback_observation(tracked: Any, observations: List[Any]) -> Optional[Any]:
    best = None
    best_iou = 0.0
    for observation in observations:
        if int(observation.class_id) != int(tracked.class_id):
            continue
        iou = bbox_iou_xyxy(tracked.bbox_xyxy, observation.bbox_xyxy)
        if iou > best_iou:
            best_iou = iou
            best = observation
    return best if best_iou >= 0.80 else None
