"""Dummy tests for ReID appearance cost helpers."""

import numpy as np

from deep_oc_sort_3d.mtmc.global_reid_cost import (
    appearance_distance_from_embeddings,
    combine_geometry_and_reid_cost,
    cosine_similarity,
)


def test_cosine_distance_identical_and_different():
    identical = appearance_distance_from_embeddings(np.asarray([1.0, 0.0]), np.asarray([2.0, 0.0]))
    orthogonal = appearance_distance_from_embeddings(np.asarray([1.0, 0.0]), np.asarray([0.0, 1.0]))

    assert cosine_similarity(np.asarray([1.0, 0.0]), np.asarray([2.0, 0.0])) == 1.0
    assert identical[0] == 0.0
    assert orthogonal[0] > identical[0]


def test_combine_geometry_and_reid_cost_and_fallback():
    total, used, reason = combine_geometry_and_reid_cost(0.5, 0.2, 0.1, use_reid=True)
    assert abs(total - 0.52) < 1e-9
    assert used
    assert reason == ""

    fallback_total, fallback_used, fallback_reason = combine_geometry_and_reid_cost(0.5, None, 0.1, use_reid=True)
    assert fallback_total == 0.5
    assert not fallback_used
    assert fallback_reason == "embedding_missing_fallback_geometry"

    disabled_total, disabled_used, disabled_reason = combine_geometry_and_reid_cost(0.5, 0.2, 0.1, use_reid=False)
    assert disabled_total == 0.5
    assert not disabled_used
    assert disabled_reason == "reid_disabled"
