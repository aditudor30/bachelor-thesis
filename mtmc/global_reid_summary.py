"""Summary helpers for global MTMC association with ReID diagnostics."""

import csv
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from deep_oc_sort_3d.mtmc.candidate_types import MTMCTrackletCandidate
from deep_oc_sort_3d.mtmc.global_summary import summarize_global_association
from deep_oc_sort_3d.mtmc.global_types import GlobalAssociationEdge, GlobalTrack


def summarize_reid_global_association(
    global_tracks: List[GlobalTrack],
    edges: List[GlobalAssociationEdge],
    candidates: List[MTMCTrackletCandidate],
) -> Dict[str, Any]:
    """Build global association summary enriched with appearance statistics."""
    summary = summarize_global_association(global_tracks, edges, candidates)
    accepted_edges = [edge for edge in edges if edge.accepted]
    rejected_edges = [edge for edge in edges if not edge.accepted]
    with_reid = [edge for edge in edges if edge.used_reid]
    missing_reid = [edge for edge in edges if not edge.used_reid]
    accepted_with_reid = [edge for edge in accepted_edges if edge.used_reid]
    accepted_without_reid = [edge for edge in accepted_edges if not edge.used_reid]
    multi_camera = [track for track in global_tracks if track.num_cameras > 1]
    summary.update(
        {
            "total_edges": len(edges),
            "accepted_edges_with_reid": len(accepted_with_reid),
            "accepted_edges_without_reid": len(accepted_without_reid),
            "pairs_with_embeddings": len(with_reid),
            "pairs_missing_embeddings": len(missing_reid),
            "mean_appearance_distance_all": _mean(_appearance_values(edges)),
            "mean_appearance_distance_accepted": _mean(_appearance_values(accepted_edges)),
            "mean_appearance_distance_rejected": _mean(_appearance_values(rejected_edges)),
            "mean_cosine_similarity_accepted": _mean(_similarity_values(accepted_edges)),
            "mean_cosine_similarity_rejected": _mean(_similarity_values(rejected_edges)),
            "per_class_multi_camera": _count_tracks_by(multi_camera, "class_name"),
            "reid_missing_reasons": _reid_missing_reasons(edges),
            "accepted_edge_temporal_relations": _count_edges_by(accepted_edges, "temporal_relation"),
            "reid_used_ratio": _ratio(len(with_reid), len(edges)),
        }
    )
    return summary


def print_reid_global_summary(summary: Dict[str, Any]) -> None:
    """Print compact ReID global association summary."""
    print("total_candidates: %s" % summary.get("total_candidates"))
    print("total_edges: %s" % summary.get("total_edges"))
    print("accepted_edges: %s" % summary.get("accepted_edges"))
    print("accepted_edges_with_reid: %s" % summary.get("accepted_edges_with_reid"))
    print("accepted_edges_without_reid: %s" % summary.get("accepted_edges_without_reid"))
    print("pairs_with_embeddings: %s" % summary.get("pairs_with_embeddings"))
    print("pairs_missing_embeddings: %s" % summary.get("pairs_missing_embeddings"))
    print("global_tracks: %s" % summary.get("global_tracks"))
    print("multi_camera_tracks: %s" % summary.get("multi_camera_tracks"))
    print("singleton_tracks: %s" % summary.get("singleton_tracks"))
    print("mean_appearance_distance_accepted: %s" % summary.get("mean_appearance_distance_accepted"))
    gt_metrics = summary.get("diagnostic_gt_metrics", {})
    if isinstance(gt_metrics, dict):
        print("global_purity_mean: %s" % gt_metrics.get("global_purity_mean"))
        print("false_merge_rate: %s" % gt_metrics.get("false_merge_rate"))
        print("fragmentation_approx: %s" % gt_metrics.get("fragmentation_approx"))
    print("per_class_tracks: %s" % json.dumps(summary.get("per_class_tracks", {}), sort_keys=True))
    print("reid_missing_reasons: %s" % json.dumps(summary.get("reid_missing_reasons", {}), sort_keys=True))


def write_reid_global_summary_json(summary: Dict[str, Any], path: Path) -> None:
    """Write ReID global summary JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")


def write_reid_global_summary_csv(summary: Dict[str, Any], path: Path) -> None:
    """Write ReID global summary as compact CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["metric", "value"])
        writer.writeheader()
        for key, value in summary.items():
            if isinstance(value, (dict, list)):
                value = json.dumps(value, sort_keys=True)
            writer.writerow({"metric": key, "value": value})


def _appearance_values(edges: List[GlobalAssociationEdge]) -> List[float]:
    return [float(edge.appearance_distance) for edge in edges if edge.appearance_distance is not None]


def _similarity_values(edges: List[GlobalAssociationEdge]) -> List[float]:
    return [float(edge.cosine_similarity) for edge in edges if edge.cosine_similarity is not None]


def _count_tracks_by(global_tracks: List[GlobalTrack], field: str) -> Dict[str, int]:
    counts = {}
    for track in global_tracks:
        key = str(getattr(track, field))
        counts[key] = counts.get(key, 0) + 1
    return counts


def _count_edges_by(edges: List[GlobalAssociationEdge], field: str) -> Dict[str, int]:
    counts = {}
    for edge in edges:
        key = str(getattr(edge, field))
        counts[key] = counts.get(key, 0) + 1
    return counts


def _reid_missing_reasons(edges: List[GlobalAssociationEdge]) -> Dict[str, int]:
    counts = {}
    for edge in edges:
        reason = str(edge.reid_missing_reason or ("used_reid" if edge.used_reid else "unknown"))
        counts[reason] = counts.get(reason, 0) + 1
    return counts


def _mean(values: List[Any]) -> Any:
    if not values:
        return None
    return float(np.mean(np.asarray(values, dtype=float)))


def _ratio(numerator: int, denominator: int) -> Any:
    if denominator <= 0:
        return None
    return float(numerator) / float(denominator)
