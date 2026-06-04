"""Diagnostic evaluation helpers for global MTMC ReID runs."""

from typing import Any, Dict, List

import numpy as np

from deep_oc_sort_3d.mtmc.global_eval import evaluate_global_tracks
from deep_oc_sort_3d.mtmc.global_types import GlobalAssociationEdge, GlobalTrack


def evaluate_reid_global_tracks(
    global_tracks: List[GlobalTrack],
    edges: List[GlobalAssociationEdge],
) -> Dict[str, Any]:
    """Evaluate global tracks and append ReID edge diagnostics."""
    metrics = evaluate_global_tracks(global_tracks)
    accepted = [edge for edge in edges if edge.accepted]
    rejected = [edge for edge in edges if not edge.accepted]
    accepted_with_reid = [edge for edge in accepted if edge.used_reid]
    high_appearance_false_merge = _false_merge_edges(accepted, min_distance=0.75)
    metrics.update(
        {
            "accepted_edges": len(accepted),
            "accepted_edges_with_reid": len(accepted_with_reid),
            "accepted_edges_without_reid": len(accepted) - len(accepted_with_reid),
            "mean_appearance_distance_accepted": _mean(
                [edge.appearance_distance for edge in accepted if edge.appearance_distance is not None]
            ),
            "mean_appearance_distance_rejected": _mean(
                [edge.appearance_distance for edge in rejected if edge.appearance_distance is not None]
            ),
            "mean_cosine_similarity_accepted": _mean(
                [edge.cosine_similarity for edge in accepted if edge.cosine_similarity is not None]
            ),
            "mean_cosine_similarity_rejected": _mean(
                [edge.cosine_similarity for edge in rejected if edge.cosine_similarity is not None]
            ),
            "false_merges_with_high_appearance_distance": len(high_appearance_false_merge),
            "per_class_appearance_effect": _per_class_appearance_effect(edges),
            "same_gt_appearance_stats": _same_different_gt_stats(edges, same=True),
            "different_gt_appearance_stats": _same_different_gt_stats(edges, same=False),
        }
    )
    return metrics


def _false_merge_edges(edges: List[GlobalAssociationEdge], min_distance: float) -> List[GlobalAssociationEdge]:
    output = []
    for edge in edges:
        if edge.appearance_distance is None or float(edge.appearance_distance) < float(min_distance):
            continue
        output.append(edge)
    return output


def _per_class_appearance_effect(edges: List[GlobalAssociationEdge]) -> Dict[str, Any]:
    values = {}
    for edge in edges:
        if edge.appearance_distance is None:
            continue
        values.setdefault(edge.class_name, {"accepted": [], "rejected": []})
        key = "accepted" if edge.accepted else "rejected"
        values[edge.class_name][key].append(float(edge.appearance_distance))
    output = {}
    for class_name, groups in values.items():
        output[class_name] = {
            "accepted_mean": _mean(groups.get("accepted", [])),
            "rejected_mean": _mean(groups.get("rejected", [])),
            "accepted_count": len(groups.get("accepted", [])),
            "rejected_count": len(groups.get("rejected", [])),
        }
    return output


def _same_different_gt_stats(edges: List[GlobalAssociationEdge], same: bool) -> Dict[str, Any]:
    # Edge-level GT labels are optional and usually absent in official/test runs.
    values = []
    for edge in edges:
        label = getattr(edge, "same_gt_object_id", None)
        if label is None or bool(label) != bool(same) or edge.appearance_distance is None:
            continue
        values.append(float(edge.appearance_distance))
    return {
        "count": len(values),
        "mean": _mean(values),
        "median": _median(values),
    }


def _mean(values: List[Any]) -> Any:
    if not values:
        return None
    return float(np.mean(np.asarray(values, dtype=float)))


def _median(values: List[Any]) -> Any:
    if not values:
        return None
    return float(np.median(np.asarray(values, dtype=float)))
