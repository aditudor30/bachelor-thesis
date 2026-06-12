import torch

from deep_oc_sort_3d.learned_association.mlp_pairwise_scorer import PairwiseMLPScorer


def test_mlp_forward_returns_one_logit_per_pair():
    model = PairwiseMLPScorer(
        input_dim=12,
        hidden_dims=[16, 8],
        dropout=0.0,
        batch_norm=False,
        layer_norm=True,
    )
    features = torch.randn(4, 12)

    logits = model(features)

    assert logits.shape == (4,)
    assert torch.isfinite(logits).all()
