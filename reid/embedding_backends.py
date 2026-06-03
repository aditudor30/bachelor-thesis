"""Pluggable ReID embedding backends."""

from typing import Any, Dict, List

import cv2
import numpy as np

from deep_oc_sort_3d.reid.reid_types import normalize_embedding_l2


class BaseEmbeddingBackend(object):
    """Base class for crop embedding backends."""

    name = "base"
    embedding_dim = 0

    def extract(self, crop: np.ndarray) -> np.ndarray:
        """Extract one embedding."""
        raise NotImplementedError

    def extract_batch(self, crops: List[np.ndarray]) -> np.ndarray:
        """Extract embeddings for a list of crops."""
        if not crops:
            return np.zeros((0, int(self.embedding_dim)), dtype=float)
        return np.vstack([self.extract(crop).reshape(1, -1) for crop in crops])


class ColorHistogramBackend(BaseEmbeddingBackend):
    """Deterministic color histogram ReID baseline."""

    name = "color_histogram"

    def __init__(self, bins_per_channel: int = 16, color_space: str = "rgb", resize: Any = (128, 256)):
        self.bins_per_channel = int(bins_per_channel)
        self.color_space = str(color_space).lower()
        self.resize = resize
        self.embedding_dim = self.bins_per_channel * 3

    def extract(self, crop: np.ndarray) -> np.ndarray:
        """Extract an L2-normalized RGB/HSV per-channel histogram."""
        if crop is None or np.asarray(crop).size == 0:
            return np.zeros((self.embedding_dim,), dtype=float)
        image = np.asarray(crop)
        if self.resize is not None:
            image = cv2.resize(image, (int(self.resize[0]), int(self.resize[1])), interpolation=cv2.INTER_LINEAR)
        if self.color_space == "hsv":
            image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        hist_parts = []
        for channel in range(3):
            values = image[:, :, channel].reshape(-1)
            hist, _edges = np.histogram(values, bins=self.bins_per_channel, range=(0, 256))
            hist_parts.append(hist.astype(float))
        embedding = np.concatenate(hist_parts, axis=0)
        return normalize_embedding_l2(embedding)


class DummyEmbeddingBackend(BaseEmbeddingBackend):
    """Small deterministic backend for tests."""

    name = "dummy"

    def __init__(self, embedding_dim: int = 8):
        self.embedding_dim = int(embedding_dim)

    def extract(self, crop: np.ndarray) -> np.ndarray:
        """Return a deterministic vector based on crop shape and mean intensity."""
        arr = np.asarray(crop, dtype=float)
        if arr.size == 0:
            base = 0.0
            shape_sum = 0.0
        else:
            base = float(np.mean(arr)) / 255.0
            shape_sum = float(sum(arr.shape)) / 1000.0
        values = np.asarray([base + shape_sum + float(index) * 0.01 for index in range(self.embedding_dim)], dtype=float)
        return normalize_embedding_l2(values)


class TorchvisionResNetBackend(BaseEmbeddingBackend):
    """Optional torchvision ResNet feature extractor without downloads."""

    name = "torchvision_resnet"

    def __init__(
        self,
        model_name: str = "resnet50",
        weights_path: Any = None,
        allow_random_weights: bool = False,
        device: str = "cpu",
    ):
        self.model_name = str(model_name)
        self.device = str(device)
        self.weights_path = weights_path
        self.allow_random_weights = bool(allow_random_weights)
        self.model, self.embedding_dim = self._build_model()

    def _build_model(self):
        try:
            import torch
            import torchvision.models as models
        except ImportError:
            raise RuntimeError("torch/torchvision are required for torchvision_resnet backend")
        if self.model_name == "resnet18":
            model = models.resnet18(weights=None)
            dim = int(model.fc.in_features)
        elif self.model_name == "resnet50":
            model = models.resnet50(weights=None)
            dim = int(model.fc.in_features)
        else:
            raise ValueError("Unsupported torchvision model: %s" % self.model_name)
        if self.weights_path not in (None, ""):
            state = torch.load(str(self.weights_path), map_location="cpu")
            model.load_state_dict(state)
        elif not self.allow_random_weights:
            raise RuntimeError("No local weights_path provided for torchvision_resnet and allow_random_weights=False")
        model.fc = torch.nn.Identity()
        model.eval()
        model.to(self.device)
        return model, dim

    def extract(self, crop: np.ndarray) -> np.ndarray:
        """Extract a normalized ResNet feature vector."""
        import torch

        image = cv2.resize(np.asarray(crop), (224, 224), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
        tensor = torch.from_numpy(image.transpose(2, 0, 1)).float().unsqueeze(0)
        tensor = tensor.to(self.device)
        with torch.no_grad():
            feature = self.model(tensor).detach().cpu().numpy().reshape(-1)
        return normalize_embedding_l2(feature)


def build_embedding_backend(config: Dict[str, Any]) -> BaseEmbeddingBackend:
    """Build an embedding backend from a simple config dictionary."""
    backend_name = str(config.get("backend", config.get("name", "color_histogram")))
    if backend_name == "dummy":
        return DummyEmbeddingBackend(embedding_dim=int(config.get("embedding_dim", 8)))
    if backend_name == "torchvision_resnet":
        try:
            return TorchvisionResNetBackend(
                model_name=str(config.get("model_name", "resnet50")),
                weights_path=config.get("weights_path"),
                allow_random_weights=bool(config.get("allow_random_weights", False)),
                device=str(config.get("device", "cpu")),
            )
        except Exception as exc:
            fallback = str(config.get("fallback_backend", "color_histogram"))
            if fallback == "color_histogram":
                print("warning: torchvision_resnet unavailable (%s); falling back to color_histogram" % str(exc))
                return ColorHistogramBackend(
                    bins_per_channel=int(config.get("bins_per_channel", 16)),
                    color_space=str(config.get("color_space", "rgb")),
                    resize=config.get("resize", (128, 256)),
                )
            raise
    return ColorHistogramBackend(
        bins_per_channel=int(config.get("bins_per_channel", 16)),
        color_space=str(config.get("color_space", "rgb")),
        resize=config.get("resize", (128, 256)),
    )

