import numpy as np
import torch

from deep_oc_sort_3d.learned_association_application.mlp_scorer_loader import score_dummy_matrix


class DummyModel(torch.nn.Module):
    def forward(self, features):
        return features[:, 0]


def test_dummy_mlp_scores_are_sigmoid_probabilities():
    scores = score_dummy_matrix(DummyModel(), np.asarray([[0.0], [2.0]], dtype=np.float32))
    assert scores.shape == (2,)
    assert abs(float(scores[0]) - 0.5) < 1e-6
    assert float(scores[1]) > float(scores[0])
