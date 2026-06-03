import numpy as np

from deep_oc_sort_3d.reid.reid_io import (
    read_reid_embeddings_jsonl,
    read_reid_embeddings_npy,
    write_reid_embeddings_jsonl,
    write_reid_embeddings_npy,
)
from deep_oc_sort_3d.reid.reid_types import ReIDEmbeddingRecord


def test_reid_jsonl_and_npy_round_trip(tmp_path):
    record = ReIDEmbeddingRecord(
        embedding_id="e1",
        subset="test",
        split="test",
        scene_name="Warehouse_023",
        camera_id="Camera_0000",
        frame_id=None,
        local_track_id=1,
        global_track_id=10,
        candidate_id="c1",
        class_id=0,
        class_name="Person",
        embedding=np.asarray([0.5, 0.5], dtype=float),
        embedding_dim=2,
        backend="dummy",
        num_crops=1,
        crop_frame_ids=[5],
        mean_confidence=0.8,
        notes="ok",
    )
    jsonl = tmp_path / "embeddings.jsonl"
    write_reid_embeddings_jsonl([record], jsonl)
    loaded = read_reid_embeddings_jsonl(jsonl)
    assert len(loaded) == 1
    np.testing.assert_allclose(loaded[0].embedding, record.embedding)

    npy = tmp_path / "embeddings.npy"
    meta = tmp_path / "embeddings.metadata.csv"
    write_reid_embeddings_npy([record], npy, meta)
    loaded_npy = read_reid_embeddings_npy(npy, meta)
    assert len(loaded_npy) == 1
    np.testing.assert_allclose(loaded_npy[0].embedding, record.embedding)
