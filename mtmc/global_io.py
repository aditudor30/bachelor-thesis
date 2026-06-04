"""CSV and JSONL I/O for global MTMC association outputs."""

import csv
import json
from pathlib import Path
from typing import Any, Dict, List

from deep_oc_sort_3d.mtmc.candidate_io import candidate_to_csv_row, candidate_to_dict
from deep_oc_sort_3d.mtmc.candidate_types import MTMCTrackletCandidate
from deep_oc_sort_3d.mtmc.global_types import (
    GlobalAssociationEdge,
    GlobalTrack,
    edge_from_dict,
    edge_to_dict,
    global_track_from_dict,
    global_track_to_dict,
)


GLOBAL_TRACK_CSV_FIELDS = [
    "global_track_id",
    "scene_name",
    "subset",
    "split",
    "class_id",
    "class_name",
    "candidate_ids_json",
    "camera_ids_json",
    "local_track_ids_json",
    "start_frame",
    "end_frame",
    "duration",
    "num_candidates",
    "num_cameras",
    "mean_confidence",
    "max_confidence",
    "trajectory_3d_sampled_json",
    "center_3d_mean_json",
    "majority_gt_object_id",
    "gt_purity",
    "num_gt_ids",
    "gt_id_counts_json",
    "notes",
]


EDGE_CSV_FIELDS = [
    "scene_name",
    "subset",
    "class_id",
    "class_name",
    "candidate_id_a",
    "candidate_id_b",
    "camera_id_a",
    "camera_id_b",
    "start_frame_a",
    "end_frame_a",
    "start_frame_b",
    "end_frame_b",
    "temporal_relation",
    "overlap_frames",
    "temporal_gap",
    "mean_3d_distance",
    "median_3d_distance",
    "min_3d_distance",
    "max_3d_distance",
    "entry_exit_distance",
    "velocity_angle_difference",
    "cost",
    "affinity",
    "accepted",
    "reject_reason",
    "geometry_cost",
    "appearance_distance",
    "cosine_similarity",
    "appearance_weight",
    "used_reid",
    "reid_missing_reason",
    "total_cost",
]


def write_global_tracks_csv(global_tracks: List[GlobalTrack], path: Path) -> None:
    """Write global tracks as compact CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=GLOBAL_TRACK_CSV_FIELDS)
        writer.writeheader()
        for track in global_tracks:
            writer.writerow(_global_track_to_csv_row(track))


def read_global_tracks_csv(path: Path) -> List[GlobalTrack]:
    """Read global tracks from compact CSV."""
    if not path.exists():
        return []
    output = []
    with path.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            output.append(global_track_from_dict(_global_track_csv_row_to_dict(row)))
    return output


def write_global_tracks_jsonl(global_tracks: List[GlobalTrack], path: Path) -> None:
    """Write global tracks as JSONL."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(global_track_to_dict(track), sort_keys=True) for track in global_tracks]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def read_global_tracks_jsonl(path: Path) -> List[GlobalTrack]:
    """Read global tracks from JSONL."""
    if not path.exists():
        return []
    output = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            output.append(global_track_from_dict(json.loads(line)))
    return output


def read_global_tracks_file(path: Path) -> List[GlobalTrack]:
    """Read global tracks from CSV or JSONL."""
    if path.suffix.lower() == ".jsonl":
        return read_global_tracks_jsonl(path)
    return read_global_tracks_csv(path)


def write_association_edges_csv(edges: List[GlobalAssociationEdge], path: Path) -> None:
    """Write association edges as CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=EDGE_CSV_FIELDS)
        writer.writeheader()
        for edge in edges:
            writer.writerow(edge_to_dict(edge))


def read_association_edges_csv(path: Path) -> List[GlobalAssociationEdge]:
    """Read association edges from CSV."""
    if not path.exists():
        return []
    output = []
    with path.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            output.append(edge_from_dict(row))
    return output


def write_association_edges_jsonl(edges: List[GlobalAssociationEdge], path: Path) -> None:
    """Write association edges as JSONL."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(edge_to_dict(edge), sort_keys=True) for edge in edges]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def read_association_edges_jsonl(path: Path) -> List[GlobalAssociationEdge]:
    """Read association edges from JSONL."""
    if not path.exists():
        return []
    output = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            output.append(edge_from_dict(json.loads(line)))
    return output


def read_association_edges_file(path: Path) -> List[GlobalAssociationEdge]:
    """Read association edges from CSV or JSONL."""
    if path.suffix.lower() == ".jsonl":
        return read_association_edges_jsonl(path)
    return read_association_edges_csv(path)


def write_candidates_with_global_ids(
    candidates: List[MTMCTrackletCandidate],
    candidate_id_to_global_track_id: Dict[str, int],
    output_csv: Path,
    output_jsonl: Path,
) -> None:
    """Write candidates with assigned global ids."""
    for candidate in candidates:
        candidate.global_track_id = candidate_id_to_global_track_id.get(candidate.candidate_id)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = list(candidate_to_csv_row(candidates[0]).keys()) if candidates else ["global_track_id"]
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for candidate in candidates:
            writer.writerow(candidate_to_csv_row(candidate))
    lines = [json.dumps(candidate_to_dict(candidate), sort_keys=True) for candidate in candidates]
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    output_jsonl.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def _global_track_to_csv_row(track: GlobalTrack) -> Dict[str, Any]:
    data = global_track_to_dict(track)
    return {
        "global_track_id": data["global_track_id"],
        "scene_name": data["scene_name"],
        "subset": data["subset"],
        "split": data["split"],
        "class_id": data["class_id"],
        "class_name": data["class_name"],
        "candidate_ids_json": json.dumps(data["candidate_ids"], sort_keys=True),
        "camera_ids_json": json.dumps(data["camera_ids"], sort_keys=True),
        "local_track_ids_json": json.dumps(data["local_track_ids"], sort_keys=True),
        "start_frame": data["start_frame"],
        "end_frame": data["end_frame"],
        "duration": data["duration"],
        "num_candidates": data["num_candidates"],
        "num_cameras": data["num_cameras"],
        "mean_confidence": data["mean_confidence"],
        "max_confidence": data["max_confidence"],
        "trajectory_3d_sampled_json": json.dumps(data["trajectory_3d_sampled"], sort_keys=True),
        "center_3d_mean_json": "" if data["center_3d_mean"] is None else json.dumps(data["center_3d_mean"]),
        "majority_gt_object_id": "" if data["majority_gt_object_id"] is None else data["majority_gt_object_id"],
        "gt_purity": "" if data["gt_purity"] is None else data["gt_purity"],
        "num_gt_ids": data["num_gt_ids"],
        "gt_id_counts_json": json.dumps(data["gt_id_counts"], sort_keys=True),
        "notes": data["notes"],
    }


def _global_track_csv_row_to_dict(row: Dict[str, str]) -> Dict[str, Any]:
    return {
        "global_track_id": _int(row.get("global_track_id"), -1),
        "scene_name": row.get("scene_name", ""),
        "subset": row.get("subset", ""),
        "split": row.get("split", ""),
        "class_id": _int(row.get("class_id"), -1),
        "class_name": row.get("class_name", ""),
        "candidate_ids": _json_value(row.get("candidate_ids_json"), []),
        "camera_ids": _json_value(row.get("camera_ids_json"), []),
        "local_track_ids": _json_value(row.get("local_track_ids_json"), []),
        "start_frame": _int(row.get("start_frame"), -1),
        "end_frame": _int(row.get("end_frame"), -1),
        "duration": _int(row.get("duration"), 0),
        "num_candidates": _int(row.get("num_candidates"), 0),
        "num_cameras": _int(row.get("num_cameras"), 0),
        "mean_confidence": _float(row.get("mean_confidence"), 0.0),
        "max_confidence": _float(row.get("max_confidence"), 0.0),
        "trajectory_3d_sampled": _json_value(row.get("trajectory_3d_sampled_json"), []),
        "center_3d_mean": _json_value(row.get("center_3d_mean_json"), None),
        "majority_gt_object_id": _optional_int(row.get("majority_gt_object_id")),
        "gt_purity": _optional_float(row.get("gt_purity")),
        "num_gt_ids": _int(row.get("num_gt_ids"), 0),
        "gt_id_counts": _json_value(row.get("gt_id_counts_json"), {}),
        "notes": row.get("notes", ""),
    }


def _json_value(value: Any, default: Any) -> Any:
    if value in (None, ""):
        return default
    return json.loads(value)


def _optional_float(value: Any) -> Any:
    if value in (None, ""):
        return None
    return float(value)


def _optional_int(value: Any) -> Any:
    if value in (None, ""):
        return None
    return int(float(value))


def _float(value: Any, default: float) -> float:
    if value in (None, ""):
        return default
    return float(value)


def _int(value: Any, default: int) -> int:
    if value in (None, ""):
        return default
    return int(float(value))
