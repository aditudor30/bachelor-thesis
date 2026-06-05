import numpy as np

from deep_oc_sort_3d.person_reid.reid_aggregation import aggregate_embeddings


def test_person_reid_aggregation_mean_normalizes():
    vectors = [np.asarray([1.0, 0.0]), np.asarray([1.0, 1.0])]
    aggregated = aggregate_embeddings(vectors, method="mean")
    assert aggregated.shape == (2,)
    assert abs(float(np.linalg.norm(aggregated)) - 1.0) < 1e-6

