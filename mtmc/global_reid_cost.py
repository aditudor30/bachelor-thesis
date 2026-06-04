"""Appearance/ReID cost helpers for global MTMC association."""

from typing import Any, Dict, Optional, Tuple

import numpy as np

from deep_oc_sort_3d.mtmc.candidate_types import MTMCTrackletCandidate
from deep_oc_sort_3d.mtmc.global_types import GlobalAssociationEdge


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> Optional[float]:
    """Compute cosine similarity or None for invalid embeddings."""
    if a is None or b is None:
        return None
    arr_a = np.asarray(a, dtype=float).reshape(-1)
    arr_b = np.asarray(b, dtype=float).reshape(-1)
    if arr_a.size == 0 or arr_b.size == 0 or arr_a.size != arr_b.size:
        return None
    denom = float(np.linalg.norm(arr_a) * np.linalg.norm(arr_b))
    if denom <= 1e-12:
        return None
    value = float(np.dot(arr_a, arr_b) / denom)
    return max(-1.0, min(1.0, value))


def appearance_distance_from_embeddings(
    emb_a: Optional[np.ndarray],
    emb_b: Optional[np.ndarray],
) -> Tuple[Optional[float], Optional[float], str]:
    """Return cosine distance, cosine similarity, and missing reason."""
    similarity = cosine_similarity(emb_a, emb_b)
    if similarity is None:
        return None, None, "missing_or_invalid_embedding"
    distance = 1.0 - float(similarity)
    distance = max(0.0, min(2.0, distance))
    return float(distance), float(similarity), ""


def combine_geometry_and_reid_cost(
    geometry_cost: float,
    appearance_distance: Optional[float],
    appearance_weight: float,
    use_reid: bool = True,
    geometry_only_fallback: bool = True,
) -> Tuple[float, bool, str]:
    """Combine geometry and appearance costs."""
    if not bool(use_reid):
        return float(geometry_cost), False, "reid_disabled"
    if appearance_distance is None:
        if geometry_only_fallback:
            return float(geometry_cost), False, "embedding_missing_fallback_geometry"
        return float("inf"), False, "embedding_missing_rejected"
    total = float(geometry_cost) + float(appearance_weight) * float(appearance_distance)
    return float(total), True, ""


def attach_reid_cost_to_edge(
    edge: GlobalAssociationEdge,
    candidate_a: MTMCTrackletCandidate,
    candidate_b: MTMCTrackletCandidate,
    embedding_lookup: Any,
    config: Dict[str, Any],
) -> GlobalAssociationEdge:
    """Attach ReID cost fields to an association edge and update acceptance."""
    use_reid = bool(config.get("use_reid", True))
    appearance_weight = float(config.get("appearance_weight", 0.10))
    require_embeddings = bool(config.get("require_embeddings_for_edge", False))
    fallback = bool(config.get("geometry_only_fallback", True)) and not require_embeddings
    geometry_cost = float(edge.geometry_cost) if edge.geometry_cost is not None else float(edge.cost)
    result_a = embedding_lookup.get_embedding(candidate_a) if embedding_lookup is not None else None
    result_b = embedding_lookup.get_embedding(candidate_b) if embedding_lookup is not None else None
    emb_a = result_a.embedding if result_a is not None and result_a.found else None
    emb_b = result_b.embedding if result_b is not None and result_b.found else None
    appearance_distance, similarity, missing = appearance_distance_from_embeddings(emb_a, emb_b)
    if missing:
        reasons = []
        if result_a is None or not result_a.found:
            reasons.append("a:%s" % (result_a.missing_reason if result_a is not None else "lookup_unavailable"))
        if result_b is None or not result_b.found:
            reasons.append("b:%s" % (result_b.missing_reason if result_b is not None else "lookup_unavailable"))
        missing = ";".join(reasons) if reasons else missing
    total_cost, used_reid, reid_missing_reason = combine_geometry_and_reid_cost(
        geometry_cost,
        appearance_distance,
        appearance_weight,
        use_reid=use_reid,
        geometry_only_fallback=fallback,
    )
    if reid_missing_reason == "":
        reid_missing_reason = missing
    elif missing:
        reid_missing_reason = "%s;%s" % (reid_missing_reason, missing)
    edge.geometry_cost = geometry_cost
    edge.appearance_distance = appearance_distance
    edge.cosine_similarity = similarity
    edge.appearance_weight = appearance_weight
    edge.used_reid = bool(used_reid)
    edge.reid_missing_reason = reid_missing_reason
    edge.total_cost = float(total_cost)
    edge.cost = float(total_cost)
    edge.affinity = 0.0 if not np.isfinite(total_cost) else 1.0 / (1.0 + float(total_cost))
    if not edge.accepted and edge.reject_reason != "cost_above_threshold":
        return edge
    threshold = _threshold_for_edge(edge, config)
    edge.accepted = bool(np.isfinite(total_cost) and float(total_cost) <= float(threshold))
    edge.reject_reason = "ok" if edge.accepted else "reid_total_cost_above_threshold"
    return edge


def _threshold_for_edge(edge: GlobalAssociationEdge, config: Dict[str, Any]) -> float:
    if edge.temporal_relation == "overlap":
        return float(config.get("cost_threshold", 1.0))
    return float(config.get("transition_cost_threshold", config.get("cost_threshold", 1.0)))
