import numpy as np

from deep_oc_sort_3d.reid_training.reid_retrieval_eval import evaluate_retrieval


def test_reid_retrieval_eval_topk_dummy():
    embeddings = np.asarray(
        [
            [1.0, 0.0],
            [0.9, 0.1],
            [0.0, 1.0],
            [0.1, 0.9],
        ],
        dtype=np.float32,
    )
    metadata = [
        {"identity_id": "a"},
        {"identity_id": "a"},
        {"identity_id": "b"},
        {"identity_id": "b"},
    ]
    result = evaluate_retrieval(embeddings, metadata, topk=[1, 2], query_chunk_size=2, max_map_queries=None)
    assert result["status"] == "ok"
    assert result["num_queries"] == 4
    assert result["top1_accuracy"] == 1.0
    assert result["top2_accuracy"] == 1.0
    assert result["mAP"] is not None
