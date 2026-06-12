"""Diagnostics and verdicts for learned-association pair datasets."""

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from deep_oc_sort_3d.learned_association.pair_dataset_io import (
    safe_float,
    write_csv_rows,
    write_json,
)


NUMERIC_FEATURES = (
    "reid_similarity",
    "reid_distance",
    "temporal_gap",
    "temporal_overlap",
    "duration_ratio",
    "start_distance_3d",
    "end_distance_3d",
    "min_endpoint_distance_3d",
    "center_mean_distance_3d",
    "spatial_distance_xy",
    "spatial_distance_z",
    "velocity_cosine",
    "velocity_difference",
    "speed_difference",
    "expected_position_error",
    "motion_consistency_score",
    "mean_conf_a",
    "mean_conf_b",
    "gt_purity_a",
    "gt_purity_b",
)


def build_dataset_diagnostics(
    fragments: Sequence[Dict[str, Any]],
    pairs: Sequence[Dict[str, Any]],
    balanced_by_split: Dict[str, Sequence[Dict[str, Any]]],
    output_root: Path,
    source_summary: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Write diagnostics, feature summaries and return the verdict payload."""
    diagnostics_dir = output_root / "diagnostics"
    features_dir = output_root / "features"
    positives = [row for row in pairs if int(row.get("same_identity") or 0) == 1]
    negatives = [row for row in pairs if int(row.get("same_identity") or 0) == 0]
    label_distribution = {
        "total_pairs": len(pairs),
        "positive_pairs": len(positives),
        "negative_pairs": len(negatives),
        "positive_ratio": len(positives) / float(max(1, len(pairs))),
        "negative_to_positive_ratio": len(negatives) / float(max(1, len(positives))),
        "hard_negatives": sum(int(row.get("hard_negative") or 0) for row in negatives),
        "balanced_train_pairs": len(balanced_by_split.get("train", [])),
        "balanced_val_pairs": len(balanced_by_split.get("val", [])),
    }
    write_json(diagnostics_dir / "label_distribution.json", label_distribution)

    camera_rows = grouped_pair_rows(pairs, "camera_pair")
    scene_rows = grouped_pair_rows(pairs, "scene_name")
    write_csv_rows(diagnostics_dir / "camera_pair_distribution.csv", camera_rows)
    write_csv_rows(diagnostics_dir / "per_camera_pair_stats.csv", camera_rows)
    write_csv_rows(diagnostics_dir / "per_scene_pair_stats.csv", scene_rows)
    write_csv_rows(diagnostics_dir / "positive_pair_examples.csv", positives[:100])
    write_csv_rows(diagnostics_dir / "negative_pair_examples.csv", negatives[:100])
    hard_negatives = [row for row in negatives if int(row.get("hard_negative") or 0) == 1]
    write_csv_rows(diagnostics_dir / "hard_negative_examples.csv", hard_negatives[:100])

    missing_rows = feature_missingness(pairs)
    write_csv_rows(features_dir / "missing_feature_report.csv", missing_rows)
    feature_summary = {
        "numeric_features": list(NUMERIC_FEATURES),
        "missingness": {row["feature"]: row["missing_rate"] for row in missing_rows},
        "positive_statistics": feature_statistics(positives),
        "negative_statistics": feature_statistics(negatives),
    }
    write_json(features_dir / "feature_summary.json", feature_summary)
    write_json(features_dir / "feature_schema.json", feature_schema(pairs))
    for split_name in ("train", "val"):
        split_pairs = [row for row in pairs if row.get("split") == split_name]
        write_csv_rows(
            features_dir / ("feature_statistics_%s.csv" % split_name),
            feature_statistics_rows(split_pairs),
        )

    warnings = collect_warnings(fragments, pairs, label_distribution, missing_rows, source_summary)
    write_json(diagnostics_dir / "warnings.json", {"warnings": warnings})
    verdict = determine_dataset_verdict(
        num_fragments=len(fragments),
        num_valid_fragments=sum(bool(row.get("valid_for_pairs")) for row in fragments),
        num_positive_pairs=len(positives),
        num_negative_pairs=len(negatives),
        missing_reid_rate=_missing_rate(missing_rows, "reid_similarity"),
        warnings=warnings,
    )
    verdict.update(
        {
            "label_distribution": label_distribution,
            "num_fragments": len(fragments),
            "num_valid_fragments": sum(bool(row.get("valid_for_pairs")) for row in fragments),
            "num_fragments_with_gt_identity": sum(
                row.get("gt_identity_id") not in (None, "", "unknown") for row in fragments
            ),
            "num_fragments_with_embedding": sum(bool(row.get("embedding_available")) for row in fragments),
        }
    )
    write_json(diagnostics_dir / "dataset_verdict.json", verdict)
    return verdict


def determine_dataset_verdict(
    num_fragments: int,
    num_valid_fragments: int,
    num_positive_pairs: int,
    num_negative_pairs: int,
    missing_reid_rate: float,
    warnings: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Choose a transparent readiness verdict."""
    reasons = []  # type: List[str]
    if num_fragments <= 0 or num_valid_fragments <= 0:
        return {
            "verdict": "person_pair_dataset_invalid_fix_required",
            "reasons": ["no_valid_fragments"],
            "ready_for_step_20b": False,
        }
    if num_positive_pairs < 100 or num_negative_pairs < 100:
        return {
            "verdict": "person_pair_dataset_too_sparse",
            "reasons": ["insufficient_positive_or_negative_pairs"],
            "ready_for_step_20b": False,
        }
    if missing_reid_rate > 0.5:
        reasons.append("high_reid_missing_rate")
    if num_valid_fragments / float(max(1, num_fragments)) < 0.25:
        reasons.append("low_valid_fragment_ratio")
    if warnings:
        reasons.extend([item for item in warnings if item not in reasons][:10])
    if reasons:
        return {
            "verdict": "person_pair_dataset_usable_with_warnings",
            "reasons": reasons,
            "ready_for_step_20b": True,
        }
    return {
        "verdict": "person_pair_dataset_ready_for_training",
        "reasons": [],
        "ready_for_step_20b": True,
    }


def grouped_pair_rows(pairs: Sequence[Dict[str, Any]], key: str) -> List[Dict[str, Any]]:
    """Summarize pair labels by a categorical key."""
    counters = defaultdict(Counter)  # type: Dict[str, Counter]
    for row in pairs:
        value = str(row.get(key) or "unknown")
        counters[value]["total_pairs"] += 1
        if int(row.get("same_identity") or 0) == 1:
            counters[value]["positive_pairs"] += 1
        else:
            counters[value]["negative_pairs"] += 1
            counters[value]["hard_negatives"] += int(row.get("hard_negative") or 0)
    result = []
    for value in sorted(counters.keys()):
        counter = counters[value]
        result.append(
            {
                key: value,
                "total_pairs": counter["total_pairs"],
                "positive_pairs": counter["positive_pairs"],
                "negative_pairs": counter["negative_pairs"],
                "hard_negatives": counter["hard_negatives"],
            }
        )
    return result


def feature_missingness(pairs: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Compute missing counts for numeric model features."""
    rows = []
    total = len(pairs)
    for feature in NUMERIC_FEATURES:
        valid = sum(safe_float(row.get(feature)) is not None for row in pairs)
        missing = total - valid
        rows.append(
            {
                "feature": feature,
                "total_count": total,
                "valid_count": valid,
                "missing_count": missing,
                "missing_rate": missing / float(max(1, total)),
            }
        )
    return rows


def feature_statistics(pairs: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Return numeric feature statistics as a dictionary."""
    result = {}
    for feature in NUMERIC_FEATURES:
        values = [safe_float(row.get(feature)) for row in pairs]
        valid = np.asarray([value for value in values if value is not None], dtype=np.float64)
        if valid.size == 0:
            result[feature] = {"count": 0, "mean": None, "median": None, "p05": None, "p95": None}
        else:
            result[feature] = {
                "count": int(valid.size),
                "mean": float(np.mean(valid)),
                "median": float(np.median(valid)),
                "p05": float(np.percentile(valid, 5)),
                "p95": float(np.percentile(valid, 95)),
            }
    return result


def feature_statistics_rows(pairs: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Return positive/negative feature statistics as CSV rows."""
    rows = []
    for label, label_name in ((1, "positive"), (0, "negative")):
        selected = [row for row in pairs if int(row.get("same_identity") or 0) == label]
        statistics = feature_statistics(selected)
        for feature, values in statistics.items():
            row = {"label": label_name, "feature": feature}
            row.update(values)
            rows.append(row)
    return rows


def feature_schema(pairs: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Describe output columns and intended model roles."""
    columns = sorted({key for row in pairs for key in row.keys() if not key.startswith("_")})
    return {
        "label_column": "same_identity",
        "id_columns": ["pair_id", "fragment_a_id", "fragment_b_id"],
        "numeric_feature_columns": list(NUMERIC_FEATURES),
        "categorical_feature_columns": ["camera_pair", "temporal_order"],
        "all_columns": columns,
    }


def collect_warnings(
    fragments: Sequence[Dict[str, Any]],
    pairs: Sequence[Dict[str, Any]],
    labels: Dict[str, Any],
    missing_rows: Sequence[Dict[str, Any]],
    source_summary: Optional[Dict[str, Any]],
) -> List[str]:
    """Collect concise actionable warnings."""
    warnings = []  # type: List[str]
    valid_ratio = sum(bool(row.get("valid_for_pairs")) for row in fragments) / float(max(1, len(fragments)))
    if valid_ratio < 0.25:
        warnings.append("low_valid_fragment_ratio")
    if int(labels.get("positive_pairs", 0)) < 100:
        warnings.append("too_few_positive_pairs")
    if int(labels.get("negative_pairs", 0)) < 100:
        warnings.append("too_few_negative_pairs")
    if _missing_rate(missing_rows, "reid_similarity") > 0.5:
        warnings.append("high_reid_missing_rate")
    train_identities = {
        row.get("gt_identity_id") for row in fragments if row.get("split") == "train" and row.get("gt_identity_id") != "unknown"
    }
    val_identities = {
        row.get("gt_identity_id") for row in fragments if row.get("split") == "val" and row.get("gt_identity_id") != "unknown"
    }
    if train_identities.intersection(val_identities):
        warnings.append("identity_leakage_between_train_and_val")
    if source_summary:
        warnings.extend(source_summary.get("warnings", []))
    return warnings


def _missing_rate(rows: Sequence[Dict[str, Any]], feature: str) -> float:
    for row in rows:
        if row.get("feature") == feature:
            return float(row.get("missing_rate") or 0.0)
    return 1.0
