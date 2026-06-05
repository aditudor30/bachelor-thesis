"""Modular Person ReID embedding backends."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from deep_oc_sort_3d.person_reid.reid_utils import l2_normalize


@dataclass
class BackendInitResult:
    """Result of backend initialization."""

    backend: Optional[Any]
    backend_name: str
    available: bool
    embedding_dim: Optional[int]
    status: str
    message: str
    weights_loaded: bool


class BasePersonReIDBackend(object):
    """Base class for Person ReID backends."""

    name = "base"
    embedding_dim = 0

    def extract_batch(self, crops: List[np.ndarray]) -> np.ndarray:
        """Extract embeddings for a crop batch."""
        if not crops:
            return np.zeros((0, int(self.embedding_dim)), dtype=float)
        return np.vstack([self.extract(crop).reshape(1, -1) for crop in crops])

    def extract(self, crop: np.ndarray) -> np.ndarray:
        """Extract one embedding."""
        raise NotImplementedError


class DummyPersonReIDBackend(BasePersonReIDBackend):
    """Deterministic dummy backend for tests only."""

    name = "dummy"

    def __init__(self, embedding_dim: int = 16) -> None:
        self.embedding_dim = int(embedding_dim)

    def extract(self, crop: np.ndarray) -> np.ndarray:
        """Return deterministic normalized features."""
        arr = np.asarray(crop, dtype=float)
        mean = float(np.mean(arr)) / 255.0 if arr.size else 0.0
        std = float(np.std(arr)) / 255.0 if arr.size else 0.0
        values = np.asarray([mean, std] + [mean + std + 0.01 * index for index in range(self.embedding_dim - 2)], dtype=float)
        return l2_normalize(values)


class TorchvisionResNetPersonBackend(BasePersonReIDBackend):
    """Torchvision ResNet feature extractor using only local weights."""

    name = "torchvision_resnet"

    def __init__(self, model_name: str, weights_path: Path, device: str, normalize_l2: bool = True) -> None:
        import torch
        import torchvision.models as models

        self.torch = torch
        self.device = str(device)
        self.normalize_l2 = bool(normalize_l2)
        if model_name == "resnet18":
            model = models.resnet18(weights=None)
        elif model_name == "resnet50":
            model = models.resnet50(weights=None)
        else:
            raise ValueError("unsupported torchvision model: %s" % model_name)
        self.embedding_dim = int(model.fc.in_features)
        state = torch.load(str(weights_path), map_location="cpu")
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        model.load_state_dict(_strip_module_prefix(state), strict=False)
        model.fc = torch.nn.Identity()
        model.eval()
        model.to(self.device)
        self.model = model

    def extract_batch(self, crops: List[np.ndarray]) -> np.ndarray:
        """Extract embeddings for a crop batch."""
        if not crops:
            return np.zeros((0, int(self.embedding_dim)), dtype=float)
        tensor = self.torch.from_numpy(np.stack([_torch_preprocess(crop, 224, 224) for crop in crops], axis=0)).float()
        tensor = tensor.to(self.device)
        with self.torch.no_grad():
            features = self.model(tensor).detach().cpu().numpy()
        if self.normalize_l2:
            features = np.vstack([l2_normalize(row).reshape(1, -1) for row in features])
        return features

    def extract(self, crop: np.ndarray) -> np.ndarray:
        """Extract one embedding."""
        return self.extract_batch([crop])[0]


class TorchreidOSNetPersonBackend(BasePersonReIDBackend):
    """Torchreid OSNet backend initialized from local weights."""

    name = "torchreid_osnet"

    def __init__(self, weights_path: Path, device: str, normalize_l2: bool = True, model_name: str = "osnet_x1_0") -> None:
        import torch
        import torchreid

        self.torch = torch
        self.device = str(device)
        self.normalize_l2 = bool(normalize_l2)
        self.model = torchreid.models.build_model(name=str(model_name), num_classes=1, pretrained=False)
        state = torch.load(str(weights_path), map_location="cpu")
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        self.model.load_state_dict(_strip_module_prefix(state), strict=False)
        self.model.eval()
        self.model.to(self.device)
        self.embedding_dim = _infer_embedding_dim(self.model, self.device, self.torch)

    def extract_batch(self, crops: List[np.ndarray]) -> np.ndarray:
        """Extract embeddings for a crop batch."""
        if not crops:
            return np.zeros((0, int(self.embedding_dim)), dtype=float)
        tensor = self.torch.from_numpy(np.stack([_torch_preprocess(crop, 128, 256) for crop in crops], axis=0)).float()
        tensor = tensor.to(self.device)
        with self.torch.no_grad():
            features = self.model(tensor).detach().cpu().numpy()
        features = features.reshape(features.shape[0], -1)
        if self.normalize_l2:
            features = np.vstack([l2_normalize(row).reshape(1, -1) for row in features])
        return features

    def extract(self, crop: np.ndarray) -> np.ndarray:
        """Extract one embedding."""
        return self.extract_batch([crop])[0]


def build_person_reid_backend(config: Dict[str, Any]) -> BackendInitResult:
    """Build a ReID backend without downloading weights."""
    name = str(config.get("name", "torchreid_osnet"))
    device = _resolve_device(str(config.get("device", "cuda")), bool(config.get("fallback_to_cpu", True)))
    normalize_l2 = bool(config.get("normalize_l2", True))
    weights_path = config.get("weights_path")
    allow_dummy = bool(config.get("allow_dummy", False))
    if name == "dummy":
        if not allow_dummy:
            return BackendInitResult(None, "dummy", False, None, "backend_unavailable", "dummy backend disabled by allow_dummy=false", False)
        backend = DummyPersonReIDBackend(embedding_dim=int(config.get("embedding_dim", 16)))
        return BackendInitResult(backend, backend.name, True, backend.embedding_dim, "ok", "dummy backend; not valid real ReID", False)
    if name == "torchreid_osnet":
        if weights_path in (None, ""):
            return BackendInitResult(None, name, False, None, "backend_unavailable", "torchreid_osnet requires local weights_path; no download attempted", False)
        try:
            backend = TorchreidOSNetPersonBackend(Path(str(weights_path)), device, normalize_l2=normalize_l2, model_name=str(config.get("model_name", "osnet_x1_0")))
            return BackendInitResult(backend, backend.name, True, backend.embedding_dim, "ok", "torchreid_osnet loaded from local weights", True)
        except Exception as exc:
            return BackendInitResult(None, name, False, None, "backend_unavailable", str(exc), False)
    if name == "botsort_reid":
        return BackendInitResult(None, name, False, None, "backend_unavailable", "botsort_reid backend placeholder: provide adapter/weights before use", False)
    if name == "torchvision_resnet":
        if weights_path in (None, ""):
            return BackendInitResult(None, name, False, None, "backend_unavailable", "torchvision_resnet requires local weights_path; no download attempted", False)
        try:
            backend = TorchvisionResNetPersonBackend(str(config.get("model_name", "resnet50")), Path(str(weights_path)), device, normalize_l2=normalize_l2)
            return BackendInitResult(backend, backend.name, True, backend.embedding_dim, "ok", "torchvision_resnet loaded from local weights", True)
        except Exception as exc:
            return BackendInitResult(None, name, False, None, "backend_unavailable", str(exc), False)
    return BackendInitResult(None, name, False, None, "backend_unavailable", "unknown backend: %s" % name, False)


def _resolve_device(device: str, fallback_to_cpu: bool) -> str:
    if device == "cuda":
        try:
            import torch

            if torch.cuda.is_available():
                return "cuda"
            if fallback_to_cpu:
                return "cpu"
        except ImportError:
            if fallback_to_cpu:
                return "cpu"
    return str(device)


def _torch_preprocess(crop: np.ndarray, width: int, height: int) -> np.ndarray:
    image = cv2.resize(np.asarray(crop), (int(width), int(height)), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
    mean = np.asarray([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
    std = np.asarray([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)
    image = (image - mean) / std
    return image.transpose(2, 0, 1)


def _strip_module_prefix(state: Any) -> Dict[str, Any]:
    if not isinstance(state, dict):
        return state
    output = {}
    for key, value in state.items():
        new_key = str(key)
        if new_key.startswith("module."):
            new_key = new_key[len("module.") :]
        output[new_key] = value
    return output


def _infer_embedding_dim(model: Any, device: str, torch_module: Any) -> int:
    dummy = torch_module.zeros((1, 3, 256, 128), dtype=torch_module.float32).to(device)
    with torch_module.no_grad():
        output = model(dummy).detach().cpu().numpy().reshape(-1)
    return int(output.size)

