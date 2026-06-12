"""Load the selected Step 20B MLP and its fitted preprocessor."""

import json
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from deep_oc_sort_3d.learned_association.mlp_pairwise_scorer import PairwiseMLPScorer


class LoadedPairScorer:
    """CPU/GPU-capable MLP inference bundle."""

    def __init__(self, model: Any, preprocessor: Any, device: Any, selected_features: List[str]) -> None:
        self.model = model
        self.preprocessor = preprocessor
        self.device = device
        self.selected_features = selected_features

    def transform(self, rows: List[Dict[str, Any]]) -> np.ndarray:
        """Apply the fitted Step 20B preprocessing object."""
        matrix = self.preprocessor.transform(rows)
        if matrix.shape[1] != len(self.selected_features):
            raise ValueError(
                "Preprocessor output dimension %d does not match selected feature count %d"
                % (matrix.shape[1], len(self.selected_features))
            )
        return matrix

    def predict_scores(self, matrix: np.ndarray, batch_size: int = 4096) -> np.ndarray:
        """Return sigmoid probabilities without retaining gradients."""
        import torch

        scores = []
        self.model.eval()
        with torch.no_grad():
            for start in range(0, len(matrix), max(1, int(batch_size))):
                tensor = torch.from_numpy(matrix[start : start + batch_size]).float().to(self.device)
                logits = self.model(tensor)
                scores.append(torch.sigmoid(logits).detach().cpu().numpy())
        return np.concatenate(scores, axis=0) if scores else np.zeros((0,), dtype=np.float32)


def load_selected_mlp(config: Dict[str, Any], device_name: Optional[str] = None) -> LoadedPairScorer:
    """Load MLP checkpoint, selected columns and fitted preprocessor."""
    import torch

    paths = config.get("paths", {})
    checkpoint_path = Path(str(paths.get("mlp_checkpoint")))
    scaler_path = Path(str(paths.get("feature_scaler_pkl")))
    features_path = Path(str(paths.get("selected_features_json")))
    if not checkpoint_path.exists():
        raise FileNotFoundError("MLP checkpoint missing: %s" % checkpoint_path)
    if not scaler_path.exists():
        raise FileNotFoundError("Feature preprocessor missing: %s" % scaler_path)
    with scaler_path.open("rb") as handle:
        preprocessor = pickle.load(handle)
    with features_path.open("r", encoding="utf-8") as handle:
        feature_payload = json.load(handle)
    selected_features = [str(value) for value in feature_payload.get("features", [])]
    checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
    input_dim = int(checkpoint.get("input_dim", len(selected_features)))
    hidden_dims = [int(value) for value in checkpoint.get("hidden_dims", [128, 64])]
    checkpoint_config = checkpoint.get("config", {})
    mlp_config = checkpoint_config.get("mlp", {}) if isinstance(checkpoint_config, dict) else {}
    model = PairwiseMLPScorer(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        dropout=float(mlp_config.get("dropout", 0.2)),
        batch_norm=bool(mlp_config.get("batch_norm", False)),
        layer_norm=bool(mlp_config.get("layer_norm", True)),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    requested = str(device_name or config.get("candidate_scoring", {}).get("device", "cuda"))
    if requested.startswith("cuda") and not torch.cuda.is_available():
        print("warning: CUDA unavailable for scorer application; using CPU")
        requested = "cpu"
    device = torch.device(requested)
    model.to(device)
    return LoadedPairScorer(model, preprocessor, device, selected_features)


def score_dummy_matrix(model: Any, matrix: np.ndarray) -> np.ndarray:
    """Small test helper for models that already return logits."""
    import torch

    with torch.no_grad():
        logits = model(torch.from_numpy(matrix).float())
        return torch.sigmoid(logits).cpu().numpy()
