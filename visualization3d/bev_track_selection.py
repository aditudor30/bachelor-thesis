"""Track selection utilities for robust BEV visualizations."""

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np


@dataclass
class BEVTrack:
    """One global track represented as BEV x-y points."""

    global_track_id: int
    class_id: Optional[int]
    class_name: Optional[str]
    frames: List[int]
    x: List[float]
    y: List[float]
    confidence: Optional[List[float]]
    length: int
    start_frame: int
    end_frame: int
    duration: int
    mean_confidence: Optional[float]


def load_bev_tracks_from_generic_csv(
    path: Union[str, Path],
    x_column: str = "center_x",
    y_column: str = "center_y",
) -> List[BEVTrack]:
    """Load BEV tracks from the generic MVP tracking export CSV."""
    groups = {}
    with Path(path).open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            track_id = _optional_int(row.get("global_track_id"))
            frame_id = _optional_int(row.get("frame_id"))
            x_value = _optional_float(row.get(x_column))
            y_value = _optional_float(row.get(y_column))
            if track_id is None or frame_id is None or x_value is None or y_value is None:
                continue
            group = groups.setdefault(
                int(track_id),
                {
                    "class_id": _optional_int(row.get("class_id")),
                    "class_name": _optional_str(row.get("class_name")),
                    "items": [],
                },
            )
            if group.get("class_id") is None:
                group["class_id"] = _optional_int(row.get("class_id"))
            if group.get("class_name") is None:
                group["class_name"] = _optional_str(row.get("class_name"))
            group["items"].append(
                {
                    "frame_id": int(frame_id),
                    "x": float(x_value),
                    "y": float(y_value),
                    "confidence": _optional_float(row.get("confidence")),
                }
            )
    return _groups_to_tracks(groups)


def load_bev_tracks_from_track1(
    path: Union[str, Path],
    scene_id: Optional[int] = None,
    x_column_name: str = "x",
    y_column_name: str = "y",
) -> List[BEVTrack]:
    """Load BEV tracks from a confirmed Track 1 text file.

    Track 1 format:
    scene_id class_id object_id frame_id x y z width length height yaw
    """
    columns = ["scene_id", "class_id", "object_id", "frame_id", "x", "y", "z", "width", "length", "height", "yaw"]
    groups = {}
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.strip().split()
            if len(parts) != len(columns):
                continue
            row = dict(zip(columns, parts))
            row_scene_id = _optional_int(row.get("scene_id"))
            if scene_id is not None and row_scene_id != int(scene_id):
                continue
            track_id = _optional_int(row.get("object_id"))
            frame_id = _optional_int(row.get("frame_id"))
            x_value = _optional_float(row.get(x_column_name))
            y_value = _optional_float(row.get(y_column_name))
            if track_id is None or frame_id is None or x_value is None or y_value is None:
                continue
            group = groups.setdefault(
                int(track_id),
                {
                    "class_id": _optional_int(row.get("class_id")),
                    "class_name": None,
                    "items": [],
                },
            )
            group["items"].append(
                {
                    "frame_id": int(frame_id),
                    "x": float(x_value),
                    "y": float(y_value),
                    "confidence": None,
                }
            )
    return _groups_to_tracks(groups)


def filter_bev_tracks(
    tracks: List[BEVTrack],
    min_track_length: int = 5,
    max_tracks: Optional[int] = 100,
    class_id: Optional[int] = None,
    class_name: Optional[str] = None,
    sort_by: str = "length",
) -> List[BEVTrack]:
    """Filter and rank tracks for visualization only."""
    output = remove_invalid_coordinate_tracks(tracks)
    output = [track for track in output if track.length >= int(min_track_length)]
    if class_id is not None:
        output = [track for track in output if track.class_id == int(class_id)]
    if class_name is not None:
        output = [track for track in output if str(track.class_name) == str(class_name)]
    output = sorted(output, key=lambda track: _sort_value(track, sort_by), reverse=True)
    if max_tracks is not None:
        output = output[: int(max_tracks)]
    return output


def remove_invalid_coordinate_tracks(tracks: List[BEVTrack]) -> List[BEVTrack]:
    """Remove tracks that contain NaN/inf coordinates or no points."""
    output = []
    for track in tracks:
        if track.length <= 0:
            continue
        x_values = np.asarray(track.x, dtype=float)
        y_values = np.asarray(track.y, dtype=float)
        if x_values.size == 0 or y_values.size == 0:
            continue
        if not np.all(np.isfinite(x_values)) or not np.all(np.isfinite(y_values)):
            continue
        output.append(track)
    return output


def _groups_to_tracks(groups: Dict[int, Dict[str, Any]]) -> List[BEVTrack]:
    tracks = []
    for track_id, group in groups.items():
        items = sorted(group.get("items", []), key=lambda item: int(item["frame_id"]))
        if not items:
            continue
        frames = [int(item["frame_id"]) for item in items]
        x_values = [float(item["x"]) for item in items]
        y_values = [float(item["y"]) for item in items]
        confidence_values = [item.get("confidence") for item in items if item.get("confidence") is not None]
        confidence = None
        mean_confidence = None
        if confidence_values:
            confidence = [float(item) for item in confidence_values]
            mean_confidence = float(np.mean(np.asarray(confidence, dtype=float)))
        start_frame = int(frames[0])
        end_frame = int(frames[-1])
        tracks.append(
            BEVTrack(
                global_track_id=int(track_id),
                class_id=group.get("class_id"),
                class_name=group.get("class_name"),
                frames=frames,
                x=x_values,
                y=y_values,
                confidence=confidence,
                length=len(frames),
                start_frame=start_frame,
                end_frame=end_frame,
                duration=int(end_frame - start_frame + 1),
                mean_confidence=mean_confidence,
            )
        )
    return tracks


def _sort_value(track: BEVTrack, sort_by: str) -> float:
    if sort_by == "mean_confidence":
        if track.mean_confidence is None:
            return -1.0
        return float(track.mean_confidence)
    if sort_by == "duration":
        return float(track.duration)
    return float(track.length)


def _optional_float(value: Any) -> Optional[float]:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _optional_int(value: Any) -> Optional[int]:
    if value in (None, ""):
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _optional_str(value: Any) -> Optional[str]:
    if value in (None, ""):
        return None
    return str(value)

