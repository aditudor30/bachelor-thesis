"""PyTorch MLP for pairwise Person association scoring."""

from typing import Any, Dict, List, Sequence


def build_mlp_from_config(input_dim: int, config: Dict[str, Any]) -> Any:
    """Build the configured PairwiseMLPScorer."""
    settings = config.get("mlp", {})
    return PairwiseMLPScorer(
        input_dim=input_dim,
        hidden_dims=[int(value) for value in settings.get("hidden_dims", [128, 64])],
        dropout=float(settings.get("dropout", 0.2)),
        batch_norm=bool(settings.get("batch_norm", False)),
        layer_norm=bool(settings.get("layer_norm", True)),
    )


class PairwiseMLPScorerBase(object):
    """Marker base retained for pickle/debug introspection."""


def _torch_module() -> Any:
    import torch.nn as nn

    return nn


class PairwiseMLPScorer(_torch_module().Module, PairwiseMLPScorerBase):
    """Small tabular MLP that returns one logit per fragment pair."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Sequence[int] = (128, 64),
        dropout: float = 0.2,
        batch_norm: bool = False,
        layer_norm: bool = True,
    ) -> None:
        super(PairwiseMLPScorer, self).__init__()
        import torch.nn as nn

        dimensions = [int(input_dim)] + [int(value) for value in hidden_dims]
        layers = []  # type: List[Any]
        for index in range(len(dimensions) - 1):
            layers.append(nn.Linear(dimensions[index], dimensions[index + 1]))
            if batch_norm:
                layers.append(nn.BatchNorm1d(dimensions[index + 1]))
            elif layer_norm:
                layers.append(nn.LayerNorm(dimensions[index + 1]))
            layers.append(nn.ReLU())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(dimensions[-1], 1))
        self.network = nn.Sequential(*layers)
        self.input_dim = int(input_dim)
        self.hidden_dims = [int(value) for value in hidden_dims]

    def forward(self, features: Any) -> Any:
        """Return logits shaped ``[batch]``."""
        return self.network(features).squeeze(-1)


def checkpoint_payload(
    model: Any,
    optimizer: Any,
    epoch: int,
    input_dim: int,
    config: Dict[str, Any],
    metrics: Dict[str, Any],
) -> Dict[str, Any]:
    """Build a self-describing checkpoint payload."""
    return {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "epoch": int(epoch),
        "input_dim": int(input_dim),
        "hidden_dims": list(model.hidden_dims),
        "config": config,
        "metrics": metrics,
    }
