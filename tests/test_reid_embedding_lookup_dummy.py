"""Dummy tests for ReID embedding lookup."""

import numpy as np

from deep_oc_sort_3d.mtmc.reid_embedding_lookup import ReIDEmbeddingLookup
from deep_oc_sort_3d.reid.reid_io import write_reid_embeddings_jsonl
from deep_oc_sort_3d.reid.reid_types import ReIDEmbeddingRecord


def _record(candidate_id, local_track_id, embedding):
    return ReIDEmbeddingRecord(
        embedding_id=str(candidate_id or "track_%s" % local_track_id),
        subset="official_val",
        split="val",
        scene_name="Warehouse_020",
        camera_id="Camera_0000",
        frame_id=None,
        local_track_id=local_track_id,
        global_track_id=None,
        candidate_id=candidate_id,
        class_id=0,
        class_name="Person",
        embedding=np.asarray(embedding, dtype=float),
        embedding_dim=2,
        backend="dummy",
        num_crops=1,
        crop_frame_ids=[0],
        mean_confidence=0.9,
        notes="dummy",
    )


def test_reid_lookup_exact_candidate_id_and_fallback(tmp_path):
    root = tmp_path / "reid"
    scene_root = root / "candidate_embeddings" / "official_val" / "Warehouse_020"
    write_reid_embeddings_jsonl(
        [
            _record("candidate_a", 10, [3.0, 0.0]),
            _record(None, 11, [0.0, 2.0]),
        ],
        scene_root / "embeddings.jsonl",
    )

    lookup = ReIDEmbeddingLookup(root)
    lookup.load_for_subset_scene("official_val", "Warehouse_020")

    exact = lookup.get_embedding_by_key("official_val", "Warehouse_020", "Camera_0000", 10, "candidate_a", 0)
    assert exact.found
    assert exact.embedding_dim == 2
    np.testing.assert_allclose(exact.embedding, np.asarray([1.0, 0.0]))

    fallback = lookup.get_embedding_by_key("official_val", "Warehouse_020", "Camera_0000", 11, "missing_id", 0)
    assert fallback.found
    np.testing.assert_allclose(fallback.embedding, np.asarray([0.0, 1.0]))

    missing = lookup.get_embedding_by_key("official_val", "Warehouse_020", "Camera_0000", 99, "missing", 0)
    assert not missing.found
    assert missing.missing_reason == "embedding_not_found"
