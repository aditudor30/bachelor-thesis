"""Top-k retrieval diagnostics for Person ReID embeddings."""

from typing import Any, Dict, List

from deep_oc_sort_3d.person_reid.reid_types import PersonEmbeddingRecord
from deep_oc_sort_3d.person_reid.reid_utils import cosine_similarity


def evaluate_topk_retrieval(records: List[PersonEmbeddingRecord], top_k_values: List[int]) -> Dict[str, Any]:
    """Evaluate top-k retrieval using GT where available."""
    labeled = [record for record in records if record.matched_gt_object_id is not None]
    if not labeled:
        return {"status": "no_gt_labels", "num_queries": 0}
    hits = {int(k): 0 for k in top_k_values}
    evaluated = 0
    for query in labeled:
        gallery = [record for record in labeled if record is not query and record.scene_name == query.scene_name]
        if not gallery:
            continue
        ranked = sorted(
            gallery,
            key=lambda item: cosine_similarity(query.embedding, item.embedding),
            reverse=True,
        )
        evaluated += 1
        for k in top_k_values:
            top = ranked[: int(k)]
            if any(item.matched_gt_object_id == query.matched_gt_object_id for item in top):
                hits[int(k)] += 1
    result = {"status": "ok", "num_queries": evaluated}
    for k in top_k_values:
        result["top%d_accuracy" % int(k)] = float(hits[int(k)]) / float(evaluated) if evaluated else None
    return result

