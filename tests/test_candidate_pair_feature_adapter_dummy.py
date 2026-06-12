import numpy as np

from deep_oc_sort_3d.learned_association_application.candidate_pair_feature_adapter import align_feature_columns


def test_feature_adapter_aligns_and_fills_missing_values():
    rows = [{"a": 1.0}, {"a": 2.0, "b": 3.0}]
    matrix, report = align_feature_columns(rows, ["a", "b"], {"b": -1.0})
    assert matrix.shape == (2, 2)
    assert np.allclose(matrix[0], [1.0, -1.0])
    assert report[1]["missing_count"] == 1
