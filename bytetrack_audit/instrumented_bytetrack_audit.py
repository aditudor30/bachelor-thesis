"""Optional mini-rerun that instruments ByteTrack lifecycle decisions."""

import json
from pathlib import Path
from typing import Any, Dict, List, Sequence

from deep_oc_sort_3d.bytetrack_audit.audit_config import instrumented_scenes, output_root, variant_tracker_settings
from deep_oc_sort_3d.bytetrack_audit.audit_io import progress_iter, write_csv, write_json
from deep_oc_sort_3d.local_tracker_benchmark.bytetrack_style_tracker import (
    ByteTrackStyleTracker,
    mark_missed,
    match_tracks,
    record_from_state,
    update_track,
)
from deep_oc_sort_3d.local_tracker_benchmark.detection_loader import (
    group_detections_by_frame,
    inventory_detection_files,
    load_camera_detections,
)
from deep_oc_sort_3d.observations.observation_io import read_observations_jsonl
from deep_oc_sort_3d.tracking.association import bbox_iou_xyxy


class InstrumentedByteTrackStyleTracker(ByteTrackStyleTracker):
    """ByteTrack-style tracker that records every lifecycle branch."""

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__(config)
        self.events = []  # type: List[Dict[str, Any]]
        self.frame_counters = []  # type: List[Dict[str, Any]]

    def update(self, frame_id: int, detections: List[Any]) -> List[Any]:
        """Run the normal update while retaining detailed counters."""
        high = [det for det in detections if det.confidence >= self.high_thresh]
        low = [det for det in detections if self.low_thresh <= det.confidence < self.high_thresh]
        removed_now = 0
        for track in self.tracks:
            if int(frame_id) - int(track.last_frame) > self.track_buffer and track.state != "dead":
                track.state = "dead"
                removed_now += 1
        active = [track for track in self.tracks if track.state != "dead" and track.misses <= self.track_buffer]
        first_matches, unmatched_tracks, unmatched_high = match_tracks(
            active, high, frame_id, self.match_thresh, use_prediction=False
        )
        records = []
        matched_track_ids = set()
        for track_index, detection_index in first_matches:
            track = active[track_index]
            detection = high[detection_index]
            previous_state = track.state
            update_track(track, detection)
            records.append(record_from_state(track, detection))
            matched_track_ids.add(track.track_id)
            self.events.append(_event(frame_id, detection, track, "matched_high", previous_state, True))
        remaining_tracks = [active[index] for index in unmatched_tracks]
        second_matches, unmatched_second_tracks, unmatched_low = match_tracks(
            remaining_tracks, low, frame_id, self.second_match_thresh, use_prediction=False
        )
        for track_index, detection_index in second_matches:
            track = remaining_tracks[track_index]
            detection = low[detection_index]
            previous_state = track.state
            update_track(track, detection)
            records.append(record_from_state(track, detection))
            matched_track_ids.add(track.track_id)
            self.events.append(_event(frame_id, detection, track, "matched_low", previous_state, True))
        unmatched_track_ids = set(remaining_tracks[index].track_id for index in unmatched_second_tracks)
        lost_now = 0
        for track in active:
            if track.track_id not in matched_track_ids and track.track_id in unmatched_track_ids:
                mark_missed(track, frame_id, self.track_buffer)
                if track.state == "lost":
                    lost_now += 1
        new_tracks = 0
        for detection_index in unmatched_high:
            detection = high[detection_index]
            if detection.confidence < self.new_track_thresh:
                self.events.append(_event(frame_id, detection, None, "unmatched_high_below_new_track", "", False))
                continue
            track = self._new_track(detection)
            new_tracks += 1
            records.append(record_from_state(track, detection))
            self.events.append(_event(frame_id, detection, track, "new_track_high", "", True))
        matched_low_indices = set(index for _track_index, index in second_matches)
        for detection_index in unmatched_low:
            if detection_index not in matched_low_indices:
                self.events.append(_event(frame_id, low[detection_index], None, "unmatched_low", "", False))
        self.tracks = [track for track in self.tracks if track.state != "dead"]
        states = _state_counts(self.tracks)
        self.frame_counters.append(
            {
                "frame_id": frame_id,
                "input_detections": len(detections),
                "high_conf_detections": len(high),
                "low_conf_detections": len(low),
                "matched_high_detections": len(first_matches),
                "matched_low_detections": len(second_matches),
                "unmatched_high_detections": len(unmatched_high),
                "unmatched_low_detections": len(unmatched_low),
                "new_tracks": new_tracks,
                "lost_tracks": lost_now,
                "removed_tracks": removed_now,
                "tentative_tracks": states.get("tentative", 0),
                "confirmed_tracks": states.get("confirmed", 0),
                "active_lost_tracks": states.get("lost", 0),
            }
        )
        return sorted(records, key=lambda item: (item.frame_id, item.track_id))


def run_instrumented_mini_rerun(config: Dict[str, Any], progress: bool = True) -> Dict[str, Any]:
    """Instrument selected cameras without writing pipeline outputs."""
    paths = config.get("paths", {})
    pipeline_root = Path(str(paths.get("yolo_pipeline_root", "")))
    observations_root = Path(str(paths.get("v2_observations_root", "")))
    selection = instrumented_scenes(config)
    inventory, warnings = inventory_detection_files(pipeline_root, selection, observations_root)
    limit = int(config.get("lifecycle_audit", {}).get("max_cameras_per_scene_for_instrumented_rerun", 3))
    inventory = _limit_cameras(inventory, limit)
    settings = variant_tracker_settings(config, "bytetrack_21c_best")
    rows = []
    event_rows = []
    frame_rows = []
    for item in progress_iter(inventory, progress, "instrumented ByteTrack cameras"):
        detections = load_camera_detections(item, float(settings.get("min_confidence_for_input", 0.001)))
        tracker = InstrumentedByteTrackStyleTracker(settings)
        records = tracker.run(group_detections_by_frame(detections))
        observations = read_observations_jsonl(Path(str(item.get("observations_path", ""))))
        exact = {(int(obs.frame_id), int(obs.detection_id)): obs for obs in observations}
        by_frame = {}
        for obs in observations:
            by_frame.setdefault(int(obs.frame_id), []).append(obs)
        exported = 0
        for record in records:
            obs = exact.get((int(record.frame_id), int(record.detection_id)))
            if obs is None:
                obs = _fallback_observation(record, by_frame.get(int(record.frame_id), []))
            if obs is not None:
                exported += 1
        prefix = {
            "subset": item.get("subset"),
            "split": item.get("split"),
            "scene_name": item.get("scene_name"),
            "camera_id": item.get("camera_id"),
        }
        for event in tracker.events:
            event_rows.append(dict(prefix, **event))
        for frame_row in tracker.frame_counters:
            frame_rows.append(dict(prefix, **frame_row))
        associated = sum(1 for event in tracker.events if event.get("associated"))
        rows.append(
            dict(
                prefix,
                input_detections=len(detections),
                detections_associated_to_any_track=associated,
                exported_local_records=exported,
                associated_but_not_exported_records=max(0, associated - exported),
                export_retention=None if associated <= 0 else float(exported) / float(associated),
                matched_high_detections=_sum(tracker.frame_counters, "matched_high_detections"),
                matched_low_detections=_sum(tracker.frame_counters, "matched_low_detections"),
                unmatched_high_detections=_sum(tracker.frame_counters, "unmatched_high_detections"),
                unmatched_low_detections=_sum(tracker.frame_counters, "unmatched_low_detections"),
                lost_tracks=_sum(tracker.frame_counters, "lost_tracks"),
                removed_tracks=_sum(tracker.frame_counters, "removed_tracks"),
            )
        )
    root = output_root(config) / "instrumented_rerun"
    write_csv(root / "instrumented_camera_summary.csv", rows)
    write_csv(root / "instrumented_lifecycle_events.csv", event_rows)
    write_csv(root / "instrumented_frame_counters.csv", frame_rows)
    summary = {"status": "ok", "cameras": len(rows), "camera_rows": rows, "warnings": warnings}
    write_json(root / "instrumented_summary.json", summary)
    return summary


def _event(
    frame_id: int,
    detection: Any,
    track: Any,
    event_type: str,
    previous_state: str,
    associated: bool,
) -> Dict[str, Any]:
    return {
        "frame_id": frame_id,
        "detection_id": detection.detection_id,
        "class_id": detection.class_id,
        "class_name": detection.class_name,
        "confidence": detection.confidence,
        "event_type": event_type,
        "associated": associated,
        "track_id": "" if track is None else track.track_id,
        "previous_state": previous_state,
        "new_state": "" if track is None else track.state,
        "track_hits": "" if track is None else track.hits,
        "track_misses": "" if track is None else track.misses,
    }


def _state_counts(tracks: Sequence[Any]) -> Dict[str, int]:
    output = {}
    for track in tracks:
        output[str(track.state)] = output.get(str(track.state), 0) + 1
    return output


def _fallback_observation(record: Any, observations: List[Any]) -> Any:
    best = None
    best_iou = 0.0
    for observation in observations:
        if int(observation.class_id) != int(record.class_id):
            continue
        iou = bbox_iou_xyxy(record.bbox_xyxy, observation.bbox_xyxy)
        if iou > best_iou:
            best_iou = iou
            best = observation
    return best if best_iou >= 0.80 else None


def _limit_cameras(rows: List[Dict[str, Any]], limit: int) -> List[Dict[str, Any]]:
    output = []
    counts = {}
    for row in rows:
        key = (str(row.get("subset")), str(row.get("scene_name")))
        if counts.get(key, 0) >= limit:
            continue
        output.append(row)
        counts[key] = counts.get(key, 0) + 1
    return output


def _sum(rows: List[Dict[str, Any]], key: str) -> int:
    return sum(int(row.get(key, 0) or 0) for row in rows)

