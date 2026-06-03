"""Simple diagnostics for ReID embedding quality."""

import random
from typing import Any, Dict, List

import numpy as np


def cosine_similarity(a, b) -> float:
    """Compute cosine similarity between two embeddings."""
    a_arr = np.asarray(a, dtype=float).reshape(-1)
    b_arr = np.asarray(b, dtype=float).reshape(-1)
    denom = float(np.linalg.norm(a_arr) * np.linalg.norm(b_arr))
    if denom <= 1e-12:
        return 0.0
    return float(np.dot(a_arr, b_arr) / denom)


def compute_pairwise_similarity_sample(
    records: List[Any],
    max_pairs: int = 10000,
    same_class_only: bool = True,
) -> Dict[str, Any]:
    """Sample pairwise cosine similarities for diagnostic use."""
    pairs = []
    count = len(records)
    for i in range(count):
        for j in range(i + 1, count):
            if same_class_only and getattr(records[i], "class_id", None) != getattr(records[j], "class_id", None):
                continue
            pairs.append((i, j))
    if len(pairs) > int(max_pairs):
        random.seed(12345)
        pairs = random.sample(pairs, int(max_pairs))
    similarities = []
    same_global = []
    different_global = []
    for i, j in pairs:
        sim = cosine_similarity(getattr(records[i], "embedding"), getattr(records[j], "embedding"))
        similarities.append(sim)
        gid_i = getattr(records[i], "global_track_id", None)
        gid_j = getattr(records[j], "global_track_id", None)
        if gid_i is not None and gid_j is not None:
            if gid_i == gid_j:
                same_global.append(sim)
            else:
                different_global.append(sim)
    return {
        "num_records": count,
        "num_pairs": len(pairs),
        "same_class_only": bool(same_class_only),
        "similarity_mean": _mean(similarities),
        "similarity_median": _median(similarities),
        "same_global_id_mean": _mean(same_global),
        "different_global_id_mean": _mean(different_global),
        "notes": "Diagnostic only; embeddings are not integrated into association in Step 14A.",
    }


def find_nearest_neighbors(
    query_record: Any,
    records: List[Any],
    top_k: int = 10,
    same_class_only: bool = True,
) -> List[Dict[str, Any]]:
    """Find nearest neighbors by cosine similarity."""
    output = []
    for record in records:
        if record is query_record:
            continue
        if same_class_only and getattr(record, "class_id", None) != getattr(query_record, "class_id", None):
            continue
        output.append(
            {
                "embedding_id": getattr(record, "embedding_id", ""),
                "similarity": cosine_similarity(getattr(query_record, "embedding"), getattr(record, "embedding")),
                "class_name": getattr(record, "class_name", ""),
                "global_track_id": getattr(record, "global_track_id", None),
            }
        )
    return sorted(output, key=lambda item: float(item["similarity"]), reverse=True)[: int(top_k)]


def _mean(values: List[float]) -> Any:
    if not values:
        return None
    return float(np.mean(np.asarray(values, dtype=float)))


def _median(values: List[float]) -> Any:
    if not values:
        return None
    return float(np.median(np.asarray(values, dtype=float)))

