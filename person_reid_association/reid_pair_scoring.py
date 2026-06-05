"""Combined geometry + ReID scoring for Person association pairs."""

from pathlib import Path
from typing import Any, Dict, List, Optional

from deep_oc_sort_3d.person_association.person_pair_scoring import score_person_pair
from deep_oc_sort_3d.person_reid_association.reid_association_io import read_csv_rows, safe_float, write_csv_rows, write_json


DEFAULT_SCORING = {
    "temporal_gap_weight": 0.15,
    "distance_weight": 0.25,
    "velocity_weight": 0.15,
    "confidence_weight": 0.10,
    "expected_position_weight": 0.15,
    "reid_weight": 0.35,
}


def score_reid_person_pair(row: Dict[str, Any], config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Score one Person candidate pair with geometry and ReID."""
    cfg = dict(DEFAULT_SCORING)
    if config:
        cfg.update(config)
    base = score_person_pair(row, cfg)
    scored = dict(base)
    similarity = safe_float(row.get("reid_similarity"), None)
    if similarity is None:
        reid_score = 1.0
        scored["reid_score_status"] = "missing_reid"
    else:
        reid_score = max(0.0, min(1.0, 1.0 - float(similarity)))
        scored["reid_score_status"] = "ok"
    geometry_score = safe_float(base.get("pair_score"), 1.0) or 1.0
    reid_weight = float(cfg.get("reid_weight", 0.35))
    geometry_weight = max(0.0, 1.0 - reid_weight)
    combined = geometry_weight * float(geometry_score) + reid_weight * float(reid_score)
    scored["geometry_pair_score"] = geometry_score
    scored["reid_score"] = reid_score
    scored["combined_pair_score"] = combined
    scored["pair_score"] = combined
    return scored


def score_reid_person_pairs(rows: List[Dict[str, Any]], config: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """Score rows."""
    return [score_reid_person_pair(row, config) for row in rows]


def summarize_reid_pair_scores(rows: List[Dict[str, Any]], threshold: Optional[float] = None) -> Dict[str, Any]:
    """Summarize scored ReID pairs."""
    with_reid = [row for row in rows if row.get("reid_score_status") == "ok"]
    passing_reid = []
    if threshold is not None:
        passing_reid = [row for row in with_reid if (safe_float(row.get("reid_similarity"), -1.0) or -1.0) >= float(threshold)]
    similarities = [row.get("reid_similarity") for row in with_reid]
    return {
        "scored_pairs": len(rows),
        "pairs_with_reid": len(with_reid),
        "pairs_missing_reid": len(rows) - len(with_reid),
        "reid_similarity_threshold": threshold,
        "pairs_passing_reid_threshold": len(passing_reid),
        "pairs_passing_reid_070": _count_at_least(similarities, 0.70),
        "pairs_passing_reid_075": _count_at_least(similarities, 0.75),
        "pairs_passing_reid_080": _count_at_least(similarities, 0.80),
        "pairs_passing_reid_082": _count_at_least(similarities, 0.82),
        "pairs_passing_reid_085": _count_at_least(similarities, 0.85),
        "same_gt_passing_reid": len([row for row in passing_reid if row.get("reid_gt_pair_label") == "same_gt"]),
        "different_gt_passing_reid": len([row for row in passing_reid if row.get("reid_gt_pair_label") == "different_gt"]),
        "combined_score_mean": _mean([row.get("combined_pair_score") for row in rows]),
        "reid_similarity_min": _percentile(similarities, 0),
        "reid_similarity_mean": _mean(similarities),
        "reid_similarity_median": _percentile(similarities, 50),
        "reid_similarity_p90": _percentile(similarities, 90),
        "reid_similarity_p95": _percentile(similarities, 95),
        "reid_similarity_p99": _percentile(similarities, 99),
        "reid_similarity_max": _percentile(similarities, 100),
    }


def load_score_and_write_reid_pairs(
    input_csv: Path,
    output_csv: Path,
    scoring_config: Dict[str, Any],
    threshold: Optional[float] = None,
    summary_json: Optional[Path] = None,
) -> Dict[str, Any]:
    """Load candidate pairs, score, and write."""
    rows, _fields = read_csv_rows(input_csv)
    scored = score_reid_person_pairs(rows, scoring_config)
    write_reid_score_rows(scored, output_csv)
    summary = summarize_reid_pair_scores(scored, threshold=threshold)
    if summary_json is not None:
        write_json(summary, summary_json)
    return summary


def write_reid_score_rows(rows: List[Dict[str, Any]], output_csv: Path) -> None:
    """Write scored rows."""
    fields = [
        "subset",
        "scene_name",
        "track_a",
        "track_b",
        "global_track_id_a",
        "global_track_id_b",
        "cameras_a",
        "cameras_b",
        "temporal_gap",
        "entry_exit_distance_3d",
        "expected_position_error",
        "velocity_angle",
        "min_mean_confidence",
        "same_gt_diagnostic",
        "reid_gt_pair_label",
        "reid_status",
        "reid_similarity",
        "geometry_pair_score",
        "reid_score",
        "combined_pair_score",
        "pair_score",
        "reid_score_status",
    ]
    write_csv_rows(rows, output_csv, fields)


def _mean(values: List[Any]) -> Optional[float]:
    numeric = [safe_float(value, None) for value in values]
    numeric = [value for value in numeric if value is not None]
    if not numeric:
        return None
    return float(sum(numeric)) / float(len(numeric))


def _count_at_least(values: List[Any], threshold: float) -> int:
    numeric = [safe_float(value, None) for value in values]
    return len([value for value in numeric if value is not None and float(value) >= float(threshold)])


def _percentile(values: List[Any], p: float) -> Optional[float]:
    numeric = sorted([safe_float(value, None) for value in values if safe_float(value, None) is not None])
    if not numeric:
        return None
    if len(numeric) == 1:
        return float(numeric[0])
    index = int(round((float(p) / 100.0) * float(len(numeric) - 1)))
    index = max(0, min(len(numeric) - 1, index))
    return float(numeric[index])
