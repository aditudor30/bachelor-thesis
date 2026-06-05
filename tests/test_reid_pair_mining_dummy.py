import numpy as np

from deep_oc_sort_3d.person_reid.reid_types import PersonEmbeddingRecord
from deep_oc_sort_3d.person_reid_association.reid_pair_mining import attach_reid_to_pairs


def _record(track_id, gt_id, embedding):
    return PersonEmbeddingRecord(
        embedding_id="emb_%s" % track_id,
        level="global_fragment",
        subset="official_val",
        split="val",
        scene_name="Warehouse_020",
        camera_id="",
        frame_id=None,
        local_track_id=None,
        global_track_id=int(track_id),
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


def test_reid_pair_mining_attaches_similarity():
    rows = [
        {
            "track_a": "official_val|Warehouse_020|0|10",
            "track_b": "official_val|Warehouse_020|0|11",
            "candidate_status": "ok",
            "same_gt_diagnostic": "true_match",
        }
    ]
    embeddings = {
        ("official_val", "Warehouse_020", "0", "10"): _record("10", 42, [1.0, 0.0]),
        ("official_val", "Warehouse_020", "0", "11"): _record("11", 42, [1.0, 0.0]),
    }
    output, summary = attach_reid_to_pairs(rows, embeddings)
    assert len(output) == 1
    assert output[0]["reid_status"] == "ok"
    assert output[0]["reid_similarity"] == 1.0
    assert output[0]["reid_gt_pair_label"] == "same_gt"
    assert summary["pairs_with_both_reid"] == 1


def test_reid_pair_mining_normalizes_numeric_track_keys():
    rows = [
        {
            "track_a": "official_val|Warehouse_020|0.0|10.0",
            "track_b": "official_val|Warehouse_020|0.0|11.0",
            "candidate_status": "ok",
        }
    ]
    embeddings = {
        ("official_val", "Warehouse_020", "0", "10"): _record("10", None, [1.0, 0.0]),
        ("official_val", "Warehouse_020", "0", "11"): _record("11", None, [1.0, 0.0]),
    }
    output, summary = attach_reid_to_pairs(rows, embeddings)
    assert output[0]["reid_status"] == "ok"
    assert summary["pairs_with_both_reid"] == 1
    assert summary["embedding_key_overlap_ratio"] == 1.0


def test_reid_pair_mining_marks_missing_embedding():
    rows = [{"track_a": "official_val|Warehouse_020|0|10", "track_b": "official_val|Warehouse_020|0|99"}]
    embeddings = {("official_val", "Warehouse_020", "0", "10"): _record("10", None, [1.0, 0.0])}
    output, summary = attach_reid_to_pairs(rows, embeddings)
    assert output[0]["reid_status"] == "missing_reid"
    assert summary["pairs_missing_reid"] == 1
