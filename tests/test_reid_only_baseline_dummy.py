import numpy as np

from deep_oc_sort_3d.learned_association.reid_only_baseline import (
    reid_predictions,
    reid_scores_from_rows,
)


def test_reid_only_thresholding_and_missing_score():
    rows = [
        {"reid_similarity": "0.85"},
        {"reid_similarity": "0.65"},
        {"reid_similarity": ""},
    ]

    scores = reid_scores_from_rows(rows)
    predictions = reid_predictions(rows, 0.75)

    assert np.allclose(scores, [0.85, 0.65, 0.0])
    assert predictions.tolist() == [1, 0, 0]
