import pytest

from deep_oc_sort_3d.learned_association.scorer_evaluator import (
    average_precision_binary,
    binary_classification_metrics,
    roc_auc_score_binary,
)


def test_binary_metrics_for_perfect_scores():
    labels = [0, 0, 1, 1]
    scores = [0.1, 0.2, 0.8, 0.9]

    metrics = binary_classification_metrics(labels, scores, 0.5)

    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["false_positive_rate"] == 0.0
    assert metrics["roc_auc"] == pytest.approx(1.0)
    assert metrics["pr_auc"] == pytest.approx(1.0)
    assert roc_auc_score_binary(labels, scores) == pytest.approx(1.0)
    assert average_precision_binary(labels, scores) == pytest.approx(1.0)
