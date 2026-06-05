"""Similarity utilities for Person ReID diagnostics."""

import random
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from deep_oc_sort_3d.person_reid.reid_types import PersonEmbeddingRecord
from deep_oc_sort_3d.person_reid.reid_utils import cosine_similarity, mean, percentile


def sample_similarity_pairs(
    records: List[PersonEmbeddingRecord],
    max_pairs: int = 200000,
    seed: int = 12345,
) -> List[Tuple[int, int]]:
    """Sample pair indices."""
    pairs = []
    count = len(records)
    for i in range(count):
        for j in range(i + 1, count):
            if records[i].scene_name != records[j].scene_name:
                continue
            pairs.append((i, j))
    if len(pairs) > int(max_pairs):
        random.seed(int(seed))
        pairs = random.sample(pairs, int(max_pairs))
    return pairs


def pair_similarity_rows(records: List[PersonEmbeddingRecord], pairs: List[Tuple[int, int]]) -> List[Dict[str, Any]]:
    """Build pair similarity rows."""
    rows = []
    for i, j in pairs:
        left = records[i]
        right = records[j]
        sim = cosine_similarity(left.embedding, right.embedding)
        label = gt_pair_label(left, right)
        rows.append(
            {
                "embedding_id_a": left.embedding_id,
                "embedding_id_b": right.embedding_id,
                "subset": left.subset,
                "scene_name": left.scene_name,
                "camera_a": left.camera_id,
                "camera_b": right.camera_id,
                "global_track_id_a": left.global_track_id,
                "global_track_id_b": right.global_track_id,
                "gt_a": left.matched_gt_object_id,
                "gt_b": right.matched_gt_object_id,
                "gt_pair_label": label,
                "same_global_id": left.global_track_id is not None and right.global_track_id is not None and left.global_track_id == right.global_track_id,
                "similarity": sim,
            }
        )
    return rows


def summarize_similarity_rows(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Summarize similarity distributions."""
    similarities = [row.get("similarity") for row in rows]
    same_gt = [row.get("similarity") for row in rows if row.get("gt_pair_label") == "same_gt"]
    diff_gt = [row.get("similarity") for row in rows if row.get("gt_pair_label") == "different_gt"]
    same_global = [row.get("similarity") for row in rows if row.get("same_global_id") in (True, "True", "true", "1")]
    diff_global = [row.get("similarity") for row in rows if row.get("same_global_id") not in (True, "True", "true", "1")]
    return {
        "num_pairs": len(rows),
        "similarity_mean": mean(similarities),
        "similarity_median": percentile(similarities, 50),
        "same_gt_count": len(same_gt),
        "different_gt_count": len(diff_gt),
        "same_gt_similarity_mean": mean(same_gt),
        "same_gt_similarity_median": percentile(same_gt, 50),
        "same_gt_similarity_p05": percentile(same_gt, 5),
        "same_gt_similarity_p95": percentile(same_gt, 95),
        "different_gt_similarity_mean": mean(diff_gt),
        "different_gt_similarity_median": percentile(diff_gt, 50),
        "different_gt_similarity_p05": percentile(diff_gt, 5),
        "different_gt_similarity_p95": percentile(diff_gt, 95),
        "separation_margin": _margin(same_gt, diff_gt),
        "same_global_id_similarity_mean": mean(same_global),
        "different_global_id_similarity_mean": mean(diff_global),
    }


def threshold_sweep(rows: List[Dict[str, Any]], thresholds: List[float]) -> List[Dict[str, Any]]:
    """Evaluate same-GT matching over similarity thresholds."""
    labeled = [row for row in rows if row.get("gt_pair_label") in ("same_gt", "different_gt")]
    output = []
    for threshold in thresholds:
        tp = fp = tn = fn = 0
        for row in labeled:
            pred = float(row.get("similarity", 0.0)) >= float(threshold)
            truth = row.get("gt_pair_label") == "same_gt"
            if pred and truth:
                tp += 1
            elif pred and not truth:
                fp += 1
            elif not pred and truth:
                fn += 1
            else:
                tn += 1
        precision = float(tp) / float(tp + fp) if (tp + fp) > 0 else None
        recall = float(tp) / float(tp + fn) if (tp + fn) > 0 else None
        fpr = float(fp) / float(fp + tn) if (fp + tn) > 0 else None
        fnr = float(fn) / float(fn + tp) if (fn + tp) > 0 else None
        output.append(
            {
                "threshold": threshold,
                "tp": tp,
                "fp": fp,
                "tn": tn,
                "fn": fn,
                "precision": precision,
                "recall": recall,
                "fpr": fpr,
                "fnr": fnr,
            }
        )
    return output


def gt_pair_label(left: PersonEmbeddingRecord, right: PersonEmbeddingRecord) -> str:
    """Return same/different/unknown GT label."""
    if left.matched_gt_object_id is None or right.matched_gt_object_id is None:
        return "unknown_gt"
    if int(left.matched_gt_object_id) == int(right.matched_gt_object_id):
        return "same_gt"
    return "different_gt"


def _margin(same_gt: List[Any], diff_gt: List[Any]) -> Optional[float]:
    same_mean = mean(same_gt)
    diff_mean = mean(diff_gt)
    if same_mean is None or diff_mean is None:
        return None
    return float(same_mean) - float(diff_mean)

