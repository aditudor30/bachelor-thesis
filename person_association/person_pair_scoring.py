"""Score mined Person fragment pairs with geometry/time/motion cues."""

from pathlib import Path
from typing import Any, Dict, List, Optional

from deep_oc_sort_3d.person_association.person_association_io import read_csv_rows, safe_float, write_csv_rows, write_json


DEFAULT_SCORING = {
    "temporal_gap_weight": 0.2,
    "distance_weight": 0.4,
    "velocity_weight": 0.2,
    "confidence_weight": 0.2,
    "expected_position_weight": 0.2,
    "max_temporal_gap": 45.0,
    "max_entry_exit_distance": 2.0,
    "max_expected_position_error": 3.0,
    "max_velocity_angle": 120.0,
}


def score_person_pair(row: Dict[str, Any], config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Score one candidate pair; lower is better."""
    cfg = dict(DEFAULT_SCORING)
    if config:
        cfg.update(config)
    scored = dict(row)
    gap_score = _ratio(row.get("temporal_gap"), cfg.get("max_temporal_gap"))
    distance_score = _ratio(row.get("entry_exit_distance_3d"), cfg.get("max_entry_exit_distance"))
    expected_score = _ratio(row.get("expected_position_error"), cfg.get("max_expected_position_error"))
    angle_score = _ratio(row.get("velocity_angle"), cfg.get("max_velocity_angle"), missing=0.5)
    conf_score = 1.0 - max(0.0, min(1.0, safe_float(row.get("min_mean_confidence"), 0.0) or 0.0))
    score = (
        float(cfg.get("temporal_gap_weight", 0.2)) * gap_score
        + float(cfg.get("distance_weight", 0.4)) * distance_score
        + float(cfg.get("expected_position_weight", 0.2)) * expected_score
        + float(cfg.get("velocity_weight", 0.2)) * angle_score
        + float(cfg.get("confidence_weight", 0.2)) * conf_score
    )
    scored["temporal_gap_score"] = gap_score
    scored["distance_score"] = distance_score
    scored["expected_position_score"] = expected_score
    scored["velocity_score"] = angle_score
    scored["confidence_score"] = conf_score
    scored["pair_score"] = float(score)
    scored["score_status"] = "scored" if row.get("candidate_status") == "ok" else "rejected_before_score"
    return scored


def score_person_pairs(rows: List[Dict[str, Any]], config: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """Score all mined pairs."""
    return [score_person_pair(row, config) for row in rows]


def summarize_pair_scores(rows: List[Dict[str, Any]], max_pair_score: Optional[float] = None) -> Dict[str, Any]:
    """Summarize pair scores and diagnostic GT labels."""
    scored = [row for row in rows if row.get("score_status") == "scored"]
    accepted = []
    if max_pair_score is not None:
        accepted = [row for row in scored if (safe_float(row.get("pair_score"), None) is not None and float(row.get("pair_score")) <= float(max_pair_score))]
    true_rows = [row for row in accepted if row.get("same_gt_diagnostic") == "true_match"]
    false_rows = [row for row in accepted if row.get("same_gt_diagnostic") == "false_match"]
    return {
        "total_pairs": len(rows),
        "scored_pairs": len(scored),
        "accepted_by_score": len(accepted),
        "accepted_true_diagnostic": len(true_rows),
        "accepted_false_diagnostic": len(false_rows),
        "score_min": _min_score(scored),
        "score_mean": _mean_score(scored),
        "score_max": _max_score(scored),
    }


def load_score_and_write_pairs(
    input_csv: Path,
    output_csv: Path,
    scoring_config: Dict[str, Any],
    max_pair_score: Optional[float] = None,
    summary_json: Optional[Path] = None,
) -> Dict[str, Any]:
    """Load candidate pairs, score them, and write outputs."""
    rows, _fields = read_csv_rows(input_csv)
    scored = score_person_pairs(rows, scoring_config)
    write_score_rows(scored, output_csv)
    summary = summarize_pair_scores(scored, max_pair_score=max_pair_score)
    if summary_json is not None:
        write_json(summary, summary_json)
    return summary


def write_score_rows(rows: List[Dict[str, Any]], output_csv: Path) -> None:
    """Write score rows with stable field order."""
    base_fields = [
        "subset",
        "scene_name",
        "class_id",
        "class_name",
        "track_a",
        "track_b",
        "global_track_id_a",
        "global_track_id_b",
        "cameras_a",
        "cameras_b",
        "start_a",
        "end_a",
        "start_b",
        "end_b",
        "rows_a",
        "rows_b",
        "temporal_gap",
        "temporal_overlap",
        "entry_exit_distance_3d",
        "expected_position_error",
        "velocity_angle",
        "min_mean_confidence",
        "same_gt_diagnostic",
        "candidate_status",
        "reject_reason",
        "temporal_gap_score",
        "distance_score",
        "expected_position_score",
        "velocity_score",
        "confidence_score",
        "pair_score",
        "score_status",
    ]
    write_csv_rows(rows, output_csv, base_fields)


def _ratio(value: Any, max_value: Any, missing: float = 1.0) -> float:
    number = safe_float(value, None)
    maximum = safe_float(max_value, None)
    if number is None or maximum is None or maximum <= 0:
        return float(missing)
    return max(0.0, min(1.0, float(number) / float(maximum)))


def _scores(rows: List[Dict[str, Any]]) -> List[float]:
    values = [safe_float(row.get("pair_score"), None) for row in rows]
    return [float(value) for value in values if value is not None]


def _min_score(rows: List[Dict[str, Any]]) -> Optional[float]:
    values = _scores(rows)
    return min(values) if values else None


def _max_score(rows: List[Dict[str, Any]]) -> Optional[float]:
    values = _scores(rows)
    return max(values) if values else None


def _mean_score(rows: List[Dict[str, Any]]) -> Optional[float]:
    values = _scores(rows)
    if not values:
        return None
    return float(sum(values)) / float(len(values))

