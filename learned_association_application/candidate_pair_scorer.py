"""Score V2 Person candidate pairs with the Step 20B MLP."""

from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from deep_oc_sort_3d.learned_association_application.candidate_pair_feature_adapter import (
    missing_feature_report,
)
from deep_oc_sort_3d.learned_association_application.mlp_scorer_loader import LoadedPairScorer
from deep_oc_sort_3d.learned_association_application.scorer_association_io import safe_float, safe_int


SCORE_FIELDS = (
    "pair_id", "subset", "scene_name", "scene_id", "class_id", "class_name",
    "fragment_a_id", "fragment_b_id", "global_track_a", "global_track_b",
    "camera_a", "camera_b", "frame_start_a", "frame_end_a", "frame_start_b", "frame_end_b",
    "temporal_gap", "temporal_overlap", "spatial_distance", "reid_similarity", "mlp_score",
    "geometry_score", "motion_score", "combined_score_optional", "same_camera_temporal_conflict",
    "temporal_overlap_conflict", "large_spatial_gap_flag", "large_temporal_gap_flag",
    "missing_reid_flag", "missing_geometry_flag", "passes_mlp_077", "passes_mlp_085",
    "passes_reid_080", "passes_reid_085", "valid_for_merge", "rejection_reason",
)


def score_candidate_pairs(
    rows: List[Dict[str, Any]],
    scorer: LoadedPairScorer,
    batch_size: int = 4096,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any], List[Dict[str, Any]]]:
    """Transform and score candidate rows with exact Step 20B preprocessing."""
    matrix = scorer.transform(rows)
    scores = scorer.predict_scores(matrix, batch_size=batch_size)
    output = []
    for row, score in zip(rows, scores):
        output.append(normalize_scored_pair(row, float(score)))
    summary = score_summary(output, matrix)
    raw_features = list(getattr(scorer.preprocessor, "numeric_features", []))
    raw_features.extend(list(getattr(scorer.preprocessor, "categorical_features", [])))
    return output, summary, missing_feature_report(rows, raw_features)


def normalize_scored_pair(row: Dict[str, Any], score: float) -> Dict[str, Any]:
    """Create the stable Step 20C candidate score schema."""
    reid = safe_float(row.get("reid_similarity"), None)
    spatial = safe_float(row.get("spatial_distance"), None)
    temporal_gap = safe_float(row.get("temporal_gap"), None)
    hard_reason = _hard_rejection_reason(row)
    result = dict(row)
    result.update(
        {
            "mlp_score": float(score),
            "geometry_score": row.get("geometry_score", row.get("geometry_pair_score", "")),
            "motion_score": row.get("motion_score", row.get("motion_consistency_score", "")),
            "combined_score_optional": row.get("combined_score", row.get("combined_pair_score", "")),
            "spatial_distance": spatial,
            "passes_mlp_077": int(score >= 0.77),
            "passes_mlp_085": int(score >= 0.85),
            "passes_reid_080": int(reid is not None and reid >= 0.80),
            "passes_reid_085": int(reid is not None and reid >= 0.85),
            "valid_for_merge": int(hard_reason == "ok"),
            "rejection_reason": hard_reason,
            "large_temporal_gap_flag": int(temporal_gap is not None and temporal_gap > 300),
        }
    )
    return result


def score_summary(rows: Sequence[Dict[str, Any]], matrix: np.ndarray) -> Dict[str, Any]:
    """Summarize coverage and MLP score distribution."""
    values = np.asarray([float(row.get("mlp_score", 0.0)) for row in rows], dtype=np.float64)
    return {
        "candidate_pairs": len(rows),
        "input_dim": int(matrix.shape[1]) if matrix.ndim == 2 else None,
        "pairs_with_reid": len([row for row in rows if not bool(row.get("missing_reid_flag"))]),
        "pairs_with_geometry": len([row for row in rows if not bool(row.get("missing_geometry_flag"))]),
        "valid_for_merge": len([row for row in rows if bool(row.get("valid_for_merge"))]),
        "score_min": float(np.min(values)) if values.size else None,
        "score_mean": float(np.mean(values)) if values.size else None,
        "score_median": float(np.median(values)) if values.size else None,
        "score_p90": float(np.percentile(values, 90)) if values.size else None,
        "score_p95": float(np.percentile(values, 95)) if values.size else None,
        "score_p99": float(np.percentile(values, 99)) if values.size else None,
        "score_max": float(np.max(values)) if values.size else None,
    }


def _hard_rejection_reason(row: Dict[str, Any]) -> str:
    if safe_int(row.get("class_id"), 0) != 0:
        return "non_person"
    if int(row.get("same_camera_temporal_conflict") or 0):
        return "same_camera_temporal_conflict"
    if int(row.get("large_spatial_gap_flag") or 0):
        return "large_spatial_gap"
    if int(row.get("large_temporal_gap_flag") or 0):
        return "large_temporal_gap"
    return "ok"
