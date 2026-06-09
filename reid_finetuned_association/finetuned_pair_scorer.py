"""Candidate pair scoring with fine-tuned Person ReID embeddings."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from deep_oc_sort_3d.person_association.person_pair_mining import (
    load_person_fragments_from_final_export,
    mine_person_candidate_pairs_with_summary,
)
from deep_oc_sort_3d.person_reid_association.reid_pair_mining import attach_reid_to_pairs, load_reid_global_embeddings
from deep_oc_sort_3d.person_reid_association.reid_pair_scoring import score_reid_person_pairs
from deep_oc_sort_3d.reid_finetuned_association.finetuned_association_io import (
    output_root_from_config,
    progress_iter,
    safe_float,
    write_csv_rows,
    write_json,
)


PAIR_SCORE_FIELDS = [
    "pair_id",
    "subset",
    "scene_name",
    "class_id",
    "fragment_a",
    "fragment_b",
    "global_track_a",
    "global_track_b",
    "camera_a",
    "camera_b",
    "frame_end_a",
    "frame_start_b",
    "temporal_gap",
    "spatial_distance",
    "motion_score",
    "geometry_score",
    "reid_similarity",
    "combined_score",
    "passes_threshold_065",
    "passes_threshold_070",
    "passes_threshold_075",
    "passes_threshold_080",
    "passes_threshold_085",
    "rejection_reason",
    "reid_status",
    "same_gt_diagnostic",
    "reid_gt_pair_label",
]


def score_finetuned_candidate_pairs_from_config(config: Dict[str, Any], show_progress: bool = True) -> Dict[str, Any]:
    """Mine geometry-compatible Person pairs and attach fine-tuned ReID scores."""
    output_root = output_root_from_config(config)
    paths = config.get("paths", {})
    final_root = Path(str(paths.get("v2_final_export_root", "output/final_mvp_exports/baseline_v2_pseudo3d_fullcam")))
    fragment_dir = output_root / "embeddings" / "fragment_embeddings"
    person_cfg = config.get("candidate_scoring", {})
    class_id = int(person_cfg.get("class_id", 0))
    fragments = load_person_fragments_from_final_export(
        final_root,
        {
            "class_id": class_id,
            "subsets": person_cfg.get("subsets"),
            "scenes": person_cfg.get("scenes"),
        },
        show_progress=show_progress,
    )
    geometry_rows, geometry_summary = mine_person_candidate_pairs_with_summary(
        fragments,
        pair_mining_config(config),
        show_progress=show_progress,
    )
    embeddings = load_reid_global_embeddings(fragment_dir, person_class_id=class_id)
    rows_with_reid, reid_summary = attach_reid_to_pairs(geometry_rows, embeddings)
    scored = score_reid_person_pairs(rows_with_reid, reid_scoring_config(config))
    output_rows = normalize_pair_score_rows(scored, config)
    output_csv = output_root / "diagnostics" / "candidate_pair_reid_scores.csv"
    write_csv_rows(output_rows, output_csv, PAIR_SCORE_FIELDS)
    threshold_diagnostics = threshold_summary(output_rows, config.get("sweep", {}).get("thresholds", [0.65, 0.70, 0.75, 0.80, 0.85]))
    write_csv_rows(threshold_diagnostics, output_root / "diagnostics" / "threshold_diagnostics.csv")
    distribution = reid_score_distribution(output_rows)
    write_json(distribution, output_root / "diagnostics" / "reid_score_distribution.json")
    summary = dict(geometry_summary)
    summary.update(reid_summary)
    summary.update(
        {
            "fragments": len(fragments),
            "candidate_pair_reid_scores_csv": str(output_csv),
            "threshold_diagnostics_csv": str(output_root / "diagnostics" / "threshold_diagnostics.csv"),
            "reid_score_distribution_json": str(output_root / "diagnostics" / "reid_score_distribution.json"),
        }
    )
    write_json(summary, output_root / "diagnostics" / "candidate_pair_scoring_summary.json")
    return summary


def pair_mining_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Translate Step 18C candidate config into existing pair mining config."""
    cfg = config.get("candidate_scoring", {})
    return {
        "class_id": int(cfg.get("class_id", 0)),
        "max_temporal_gap": int(cfg.get("max_temporal_gap", 300)),
        "max_entry_exit_distance": float(cfg.get("max_spatial_distance", 12.0)),
        "max_expected_position_error": float(cfg.get("max_expected_position_error", cfg.get("max_spatial_distance", 12.0))),
        "max_velocity_angle": float(cfg.get("max_velocity_angle", 140.0)),
        "forbid_same_camera_temporal_overlap": bool(cfg.get("forbid_same_camera_temporal_overlap", True)),
        "include_rejected": False,
        "store_rejected_pairs": False,
    }


def reid_scoring_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Return combined geometry/ReID scoring weights."""
    cfg = config.get("candidate_scoring", {})
    return {
        "temporal_gap_weight": float(cfg.get("temporal_gap_weight", 0.15)),
        "distance_weight": float(cfg.get("distance_weight", 0.25)),
        "velocity_weight": float(cfg.get("velocity_weight", 0.15)),
        "confidence_weight": float(cfg.get("confidence_weight", 0.10)),
        "expected_position_weight": float(cfg.get("expected_position_weight", 0.15)),
        "reid_weight": float(cfg.get("reid_weight", 0.35)),
        "max_temporal_gap": float(cfg.get("max_temporal_gap", 300)),
        "max_entry_exit_distance": float(cfg.get("max_spatial_distance", 12.0)),
        "max_expected_position_error": float(cfg.get("max_expected_position_error", cfg.get("max_spatial_distance", 12.0))),
        "max_velocity_angle": float(cfg.get("max_velocity_angle", 140.0)),
    }


def normalize_pair_score_rows(rows: List[Dict[str, Any]], config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Normalize existing scored rows into Step 18C diagnostic schema."""
    thresholds = [float(value) for value in config.get("sweep", {}).get("thresholds", [0.65, 0.70, 0.75, 0.80, 0.85])]
    output: List[Dict[str, Any]] = []
    for index, row in enumerate(rows):
        similarity = safe_float(row.get("reid_similarity"), None)
        normalized = {
            "pair_id": "pair_%08d" % index,
            "subset": row.get("subset", ""),
            "scene_name": row.get("scene_name", ""),
            "class_id": row.get("class_id", 0),
            "fragment_a": row.get("track_a", ""),
            "fragment_b": row.get("track_b", ""),
            "global_track_a": row.get("global_track_id_a", ""),
            "global_track_b": row.get("global_track_id_b", ""),
            "camera_a": row.get("cameras_a", ""),
            "camera_b": row.get("cameras_b", ""),
            "frame_end_a": row.get("end_a", ""),
            "frame_start_b": row.get("start_b", ""),
            "temporal_gap": row.get("temporal_gap", ""),
            "spatial_distance": row.get("entry_exit_distance_3d", ""),
            "motion_score": row.get("velocity_score", ""),
            "geometry_score": row.get("geometry_pair_score", row.get("pair_score", "")),
            "reid_similarity": "" if similarity is None else similarity,
            "combined_score": row.get("combined_pair_score", ""),
            "rejection_reason": row.get("reject_reason", ""),
            "reid_status": row.get("reid_status", ""),
            "same_gt_diagnostic": row.get("same_gt_diagnostic", ""),
            "reid_gt_pair_label": row.get("reid_gt_pair_label", ""),
        }
        for threshold in thresholds:
            normalized["passes_threshold_%03d" % int(round(threshold * 100.0))] = "1" if similarity is not None and similarity >= threshold else "0"
        output.append(normalized)
    return output


def threshold_summary(rows: List[Dict[str, Any]], thresholds: List[Any]) -> List[Dict[str, Any]]:
    """Summarize pair counts per ReID threshold."""
    output = []
    for threshold in [float(value) for value in thresholds]:
        key = "passes_threshold_%03d" % int(round(threshold * 100.0))
        passing = [row for row in rows if str(row.get(key, "")) == "1"]
        output.append(
            {
                "threshold": threshold,
                "candidate_pairs": len(rows),
                "passing_pairs": len(passing),
                "same_gt": len([row for row in passing if row.get("reid_gt_pair_label") == "same_gt"]),
                "different_gt": len([row for row in passing if row.get("reid_gt_pair_label") == "different_gt"]),
                "unknown_gt": len([row for row in passing if row.get("reid_gt_pair_label") not in ("same_gt", "different_gt")]),
                "mean_similarity": _mean([row.get("reid_similarity") for row in passing]),
            }
        )
    return output


def reid_score_distribution(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute simple ReID score distribution stats."""
    values = sorted([safe_float(row.get("reid_similarity"), None) for row in rows if safe_float(row.get("reid_similarity"), None) is not None])
    return {
        "num_pairs": len(rows),
        "num_pairs_with_reid": len(values),
        "min": _percentile(values, 0),
        "mean": _mean(values),
        "median": _percentile(values, 50),
        "p75": _percentile(values, 75),
        "p90": _percentile(values, 90),
        "p95": _percentile(values, 95),
        "p99": _percentile(values, 99),
        "max": _percentile(values, 100),
    }


def _mean(values: List[Any]) -> Optional[float]:
    numeric = [safe_float(value, None) for value in values]
    numeric = [value for value in numeric if value is not None]
    if not numeric:
        return None
    return float(sum(numeric)) / float(len(numeric))


def _percentile(values: List[Any], percentile: float) -> Optional[float]:
    numeric = sorted([safe_float(value, None) for value in values if safe_float(value, None) is not None])
    if not numeric:
        return None
    if len(numeric) == 1:
        return float(numeric[0])
    index = int(round(float(percentile) / 100.0 * float(len(numeric) - 1)))
    index = max(0, min(len(numeric) - 1, index))
    return float(numeric[index])
