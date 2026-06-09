import numpy as np

from deep_oc_sort_3d.reid_finetuned_association.fragment_embedding_builder import (
    aggregate_crop_embeddings_to_fragments,
    l2_normalize_vector,
    mean_pool_l2,
)


def test_mean_pool_l2_normalizes_vectors():
    pooled = mean_pool_l2([np.asarray([1.0, 0.0]), np.asarray([1.0, 1.0])])
    assert pooled is not None
    assert abs(float(np.linalg.norm(pooled)) - 1.0) < 1e-6
    assert pooled[0] > pooled[1]


def test_fragment_aggregation_groups_by_global_track():
    embeddings = np.asarray([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]], dtype=np.float32)
    rows = [
        {"subset": "test", "scene_name": "Warehouse_023", "class_id": "0", "class_name": "Person", "global_track_id": "10", "frame_id": "1", "camera_id": "Camera_0000", "valid_embedding": "1"},
        {"subset": "test", "scene_name": "Warehouse_023", "class_id": "0", "class_name": "Person", "global_track_id": "10", "frame_id": "2", "camera_id": "Camera_0001", "valid_embedding": "1"},
        {"subset": "test", "scene_name": "Warehouse_023", "class_id": "0", "class_name": "Person", "global_track_id": "20", "frame_id": "1", "camera_id": "Camera_0000", "valid_embedding": "1"},
    ]
    matrix, index_rows, records = aggregate_crop_embeddings_to_fragments(embeddings, rows)
    assert matrix.shape == (2, 2)
    assert len(index_rows) == 2
    assert len(records) == 2
    assert abs(float(np.linalg.norm(matrix[0])) - 1.0) < 1e-6
