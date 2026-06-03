import numpy as np

from deep_oc_sort_3d.reid.reid_types import (
    ReIDEmbeddingRecord,
    normalize_embedding_l2,
    reid_embedding_from_dict,
    reid_embedding_to_dict,
)


def test_reid_embedding_to_from_dict_and_l2_normalization():
    embedding = normalize_embedding_l2(np.asarray([3.0, 4.0], dtype=float))
    np.testing.assert_allclose(np.linalg.norm(embedding), 1.0)
    record = ReIDEmbeddingRecord(
        embedding_id="e1",
        subset="official_val",
        split="val",
        scene_name="Warehouse_020",
        camera_id="Camera_0000",
        frame_id=None,
        local_track_id=1,
        global_track_id=2,
        candidate_id="c1",
        class_id=0,
        class_name="Person",
        embedding=embedding,
        embedding_dim=2,
        backend="dummy",
        num_crops=2,
        crop_frame_ids=[0, 1],
        mean_confidence=0.9,
        notes="ok",
    )
    data = reid_embedding_to_dict(record)
    loaded = reid_embedding_from_dict(data)
    assert loaded.embedding_id == "e1"
    np.testing.assert_allclose(loaded.embedding, embedding)

