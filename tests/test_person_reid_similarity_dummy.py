import numpy as np

from deep_oc_sort_3d.person_reid.reid_similarity import pair_similarity_rows, sample_similarity_pairs, summarize_similarity_rows
from deep_oc_sort_3d.person_reid.reid_types import PersonEmbeddingRecord


def _record(embedding_id, gt_id, embedding):
    return PersonEmbeddingRecord(
        embedding_id=embedding_id,
        level="global_fragment",
        subset="official_val",
        split="val",
        scene_name="Warehouse_020",
        camera_id="",
        frame_id=None,
        local_track_id=None,
        global_track_id=int(embedding_id),
        class_id=0,
        class_name="Person",
        embedding=np.asarray(embedding, dtype=float),
        embedding_dim=2,
        backend="dummy",
        num_crops=1,
        crop_ids=[],
        frame_ids=[],
        mean_confidence=0.9,
        matched_gt_object_id=gt_id,
        notes="",
    )


def test_person_reid_similarity_same_gt_higher():
    records = [
        _record("1", 10, [1.0, 0.0]),
        _record("2", 10, [0.9, 0.1]),
        _record("3", 11, [0.0, 1.0]),
    ]
    rows = pair_similarity_rows(records, sample_similarity_pairs(records, max_pairs=10))
    summary = summarize_similarity_rows(rows)
    assert summary["same_gt_similarity_mean"] > summary["different_gt_similarity_mean"]

