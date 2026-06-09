"""Retrieval metrics for ReID embeddings."""

from typing import Any, Dict, List, Optional

import numpy as np


def evaluate_retrieval(
    embeddings: np.ndarray,
    metadata: List[Dict[str, Any]],
    topk: List[int],
    query_chunk_size: int = 512,
    max_map_queries: Optional[int] = 5000,
) -> Dict[str, Any]:
    """Evaluate top-k retrieval and simplified mAP with chunked similarity.

    The validation split can contain many crops, so this function avoids
    materializing the full NxN similarity matrix at once.
    """
    if embeddings.size == 0 or not metadata:
        return {"status": "empty", "num_queries": 0}
    matrix = _l2_normalize_matrix(embeddings)
    labels = np.asarray([str(row.get("identity_id", "")) for row in metadata])
    num_items = int(len(metadata))
    max_k = max([int(k) for k in topk]) if topk else 1
    results: Dict[str, Any] = {"status": "ok", "num_queries": int(len(metadata))}
    topk_hits: Dict[int, int] = {int(k): 0 for k in topk}
    average_precisions: List[float] = []
    map_query_indices = _map_query_indices(num_items, max_map_queries)
    map_query_set = set(map_query_indices)
    chunk = max(1, int(query_chunk_size))
    for start in range(0, num_items, chunk):
        end = min(num_items, start + chunk)
        similarity = np.dot(matrix[start:end], matrix.T)
        for row_index, global_index in enumerate(range(start, end)):
            similarity[row_index, global_index] = -np.inf
        local_top = min(max_k, max(1, num_items - 1))
        top_order = np.argpartition(-similarity, local_top - 1, axis=1)[:, :local_top]
        top_scores = np.take_along_axis(similarity, top_order, axis=1)
        top_order_sorted = np.take_along_axis(top_order, np.argsort(-top_scores, axis=1), axis=1)
        for row_index, global_index in enumerate(range(start, end)):
            ordered = top_order_sorted[row_index]
            for k in topk:
                selected = ordered[: int(k)]
                if np.any(labels[selected] == labels[global_index]):
                    topk_hits[int(k)] += 1
            if global_index in map_query_set:
                full_order = np.argsort(-similarity[row_index])
                average_precision = _average_precision_for_query(full_order, labels, global_index)
                if average_precision is not None:
                    average_precisions.append(float(average_precision))
    for k in topk:
        results["top%d_accuracy" % int(k)] = float(topk_hits[int(k)]) / float(num_items)
    results["mAP"] = float(np.mean(average_precisions)) if average_precisions else None
    results["mAP_num_queries"] = len(average_precisions)
    return results


def _average_precision_for_query(order: np.ndarray, labels: np.ndarray, query_index: int) -> Optional[float]:
    positives = labels[order] == labels[query_index]
    if positives.sum() <= 0:
        return None
    precision_sum = 0.0
    hit_count = 0
    for rank, is_positive in enumerate(positives, start=1):
        if is_positive:
            hit_count += 1
            precision_sum += float(hit_count) / float(rank)
    return precision_sum / float(max(1, positives.sum()))


def _map_query_indices(num_items: int, max_map_queries: Optional[int]) -> List[int]:
    if max_map_queries is None or num_items <= int(max_map_queries):
        return list(range(num_items))
    values = np.linspace(0, num_items - 1, int(max_map_queries))
    return sorted(set([int(round(value)) for value in values]))


def _l2_normalize_matrix(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms <= 1e-12] = 1.0
    return (matrix / norms).astype(np.float32)
