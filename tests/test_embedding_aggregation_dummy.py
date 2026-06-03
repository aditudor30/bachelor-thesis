import numpy as np

from deep_oc_sort_3d.reid.embedding_aggregation import aggregate_crop_embeddings


def test_embedding_aggregation_mean_and_median():
    embeddings = [np.asarray([1.0, 0.0]), np.asarray([0.0, 1.0])]
    mean_embedding = aggregate_crop_embeddings(embeddings, method="mean")
    median_embedding = aggregate_crop_embeddings(embeddings, method="median")
    assert mean_embedding is not None
    assert median_embedding is not None
    np.testing.assert_allclose(np.linalg.norm(mean_embedding), 1.0)
    np.testing.assert_allclose(np.linalg.norm(median_embedding), 1.0)

