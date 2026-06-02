"""Summary utilities for global MTMC association."""

import csv
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from deep_oc_sort_3d.mtmc.candidate_types import MTMCTrackletCandidate
from deep_oc_sort_3d.mtmc.global_eval import evaluate_global_tracks
from deep_oc_sort_3d.mtmc.global_types import GlobalAssociationEdge, GlobalTrack


def summarize_global_association(
    global_tracks: List[GlobalTrack],
    edges: List[GlobalAssociationEdge],
    candidates: List[MTMCTrackletCandidate],
) -> Dict[str, Any]:
    """Build a compact summary for one global association run."""
    accepted_edges = [edge for edge in edges if edge.accepted]
    rejected_edges = [edge for edge in edges if not edge.accepted]
    multi_camera = [track for track in global_tracks if track.num_cameras > 1]
    singleton = [track for track in global_tracks if track.num_cameras <= 1]
    summary = {
        "total_candidates": len(candidates),
        "accepted_edges": len(accepted_edges),
        "rejected_edges": len(rejected_edges),
        "global_tracks": len(global_tracks),
        "singleton_tracks": len(singleton),
        "multi_camera_tracks": len(multi_camera),
        "per_class_tracks": _count_tracks_by(global_tracks, "class_name"),
        "per_class_multi_camera_tracks": _count_tracks_by(multi_camera, "class_name"),
        "per_camera_participation": _camera_participation(global_tracks),
        "edge_reject_reasons": _edge_reject_reasons(edges),
        "mean_cost_accepted": _mean([edge.cost for edge in accepted_edges]),
        "median_cost_accepted": _median([edge.cost for edge in accepted_edges]),
        "diagnostic_gt_metrics": evaluate_global_tracks(global_tracks),
    }
    return summary


def print_global_summary(summary: Dict[str, Any]) -> None:
    """Print compact global association summary."""
    print("total_candidates: %s" % summary.get("total_candidates"))
    print("accepted_edges: %s" % summary.get("accepted_edges"))
    print("rejected_edges: %s" % summary.get("rejected_edges"))
    print("global_tracks: %s" % summary.get("global_tracks"))
    print("singleton_tracks: %s" % summary.get("singleton_tracks"))
    print("multi_camera_tracks: %s" % summary.get("multi_camera_tracks"))
    print("per_class_tracks: %s" % json.dumps(summary.get("per_class_tracks", {}), sort_keys=True))
    print("edge_reject_reasons: %s" % json.dumps(summary.get("edge_reject_reasons", {}), sort_keys=True))
    gt_metrics = summary.get("diagnostic_gt_metrics", {})
    if isinstance(gt_metrics, dict):
        print("global_purity_mean: %s" % gt_metrics.get("global_purity_mean"))
        print("false_merge_rate: %s" % gt_metrics.get("false_merge_rate"))
        print("fragmentation_approx: %s" % gt_metrics.get("fragmentation_approx"))


def write_global_summary_json(summary: Dict[str, Any], path: Path) -> None:
    """Write summary JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")


def write_global_summary_csv(summary: Dict[str, Any], path: Path) -> None:
    """Write summary as compact metric CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["metric", "value"])
        writer.writeheader()
        for key, value in summary.items():
            if isinstance(value, (dict, list)):
                value = json.dumps(value, sort_keys=True)
            writer.writerow({"metric": key, "value": value})


def _count_tracks_by(global_tracks: List[GlobalTrack], field: str) -> Dict[str, int]:
    counts = {}
    for track in global_tracks:
        key = str(getattr(track, field))
        counts[key] = counts.get(key, 0) + 1
    return counts


def _camera_participation(global_tracks: List[GlobalTrack]) -> Dict[str, int]:
    counts = {}
    for track in global_tracks:
        for camera_id in track.camera_ids:
            counts[str(camera_id)] = counts.get(str(camera_id), 0) + 1
    return counts


def _edge_reject_reasons(edges: List[GlobalAssociationEdge]) -> Dict[str, int]:
    counts = {}
    for edge in edges:
        reason = "ok" if edge.accepted else str(edge.reject_reason)
        counts[reason] = counts.get(reason, 0) + 1
    return counts


def _mean(values: List[Any]) -> Any:
    if not values:
        return None
    return float(np.mean(np.asarray(values, dtype=float)))


def _median(values: List[Any]) -> Any:
    if not values:
        return None
    return float(np.median(np.asarray(values, dtype=float)))
