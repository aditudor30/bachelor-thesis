"""Optional local appearance adapters for Person detections."""

from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

from deep_oc_sort_3d.data.frame_io import safe_read_video_frame
from deep_oc_sort_3d.person_reid.crop_extraction import crop_image_xyxy
from deep_oc_sort_3d.local_tracker_benchmark.tracker_input_types import BenchmarkDetection


class FineTunedOSNetBackend:
    """Load the SmartSpaces fine-tuned OSNet checkpoint for inference only."""

    name = "torchreid_osnet_finetuned"

    def __init__(self, weights_path: Path, device: str = "cuda") -> None:
        import torch
        import torch.nn.functional as functional
        import torchreid

        self.torch = torch
        self.functional = functional
        self.device = _resolve_device(device, torch)
        checkpoint = torch.load(str(weights_path), map_location="cpu")
        if not isinstance(checkpoint, dict):
            raise ValueError("Fine-tuned OSNet checkpoint must be a mapping")
        state = checkpoint.get("model_state_dict")
        if not isinstance(state, dict):
            raise ValueError("Fine-tuned OSNet checkpoint is missing model_state_dict")
        checkpoint_config = checkpoint.get("config", {})
        model_config = checkpoint_config.get("model", {}) if isinstance(checkpoint_config, dict) else {}
        metrics = checkpoint.get("metrics", {})
        num_classes = int(metrics.get("num_classes", 1)) if isinstance(metrics, dict) else 1
        architecture = str(model_config.get("architecture", "osnet_x1_0"))
        self.input_height = int(model_config.get("input_height", 256))
        self.input_width = int(model_config.get("input_width", 128))
        self.model = torchreid.models.build_model(
            name=architecture,
            num_classes=max(1, num_classes),
            pretrained=False,
            loss="triplet",
        )
        compatible = _compatible_state_dict(self.model, _strip_module_prefix(state))
        if not compatible:
            raise ValueError("No compatible tensors found in fine-tuned OSNet checkpoint")
        self.model.load_state_dict(compatible, strict=False)
        self.model.eval()
        self.model.to(self.device)
        self.embedding_dim = self._infer_embedding_dim()
        self.weights_loaded = True

    def extract_batch(self, crops: List[np.ndarray]) -> np.ndarray:
        """Extract L2-normalized embeddings from RGB crops."""
        if not crops:
            return np.zeros((0, int(self.embedding_dim)), dtype=np.float32)
        arrays = [self._preprocess(crop) for crop in crops]
        tensor = self.torch.from_numpy(np.stack(arrays, axis=0)).float().to(self.device)
        with self.torch.no_grad():
            output = self.model(tensor)
            if isinstance(output, tuple) or isinstance(output, list):
                output = output[-1]
            if len(output.shape) > 2:
                output = output.reshape(output.shape[0], -1)
            output = self.functional.normalize(output, p=2, dim=1)
        return output.detach().cpu().numpy().astype(np.float32)

    def _preprocess(self, crop: np.ndarray) -> np.ndarray:
        image = cv2.resize(
            np.asarray(crop),
            (self.input_width, self.input_height),
            interpolation=cv2.INTER_LINEAR,
        ).astype(np.float32) / 255.0
        mean = np.asarray([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
        std = np.asarray([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)
        return ((image - mean) / std).transpose(2, 0, 1)

    def _infer_embedding_dim(self) -> int:
        dummy = self.torch.zeros(
            (1, 3, self.input_height, self.input_width),
            dtype=self.torch.float32,
        ).to(self.device)
        with self.torch.no_grad():
            output = self.model(dummy)
            if isinstance(output, tuple) or isinstance(output, list):
                output = output[-1]
            return int(output.reshape(1, -1).shape[1])


class PersonReIDAdapter:
    """Extract Person embeddings lazily per frame and leave non-Person untouched."""

    def __init__(self, backend: Any, video_path: Path, batch_size: int = 128) -> None:
        self.backend = backend
        self.video_path = Path(video_path)
        self.batch_size = int(batch_size)

    def attach(self, detections: List[BenchmarkDetection]) -> Dict[str, Any]:
        """Attach embeddings in-place, reading each requested frame once."""
        by_frame = {}  # type: Dict[int, List[BenchmarkDetection]]
        for detection in detections:
            if detection.class_id == 0:
                by_frame.setdefault(detection.frame_id, []).append(detection)
        embedded = 0
        missing = 0
        for frame_id, frame_detections in by_frame.items():
            image = safe_read_video_frame(self.video_path, frame_id)
            if image is None:
                missing += len(frame_detections)
                continue
            valid = []
            crops = []
            for detection in frame_detections:
                crop = crop_image_xyxy(image, detection.bbox_xyxy, padding_ratio=0.10)
                if crop is None:
                    missing += 1
                    continue
                valid.append(detection)
                crops.append(crop)
            for start in range(0, len(crops), max(1, self.batch_size)):
                embeddings = self.backend.extract_batch(crops[start : start + self.batch_size])
                for detection, embedding in zip(valid[start : start + self.batch_size], embeddings):
                    detection.embedding = np.asarray(embedding, dtype=np.float32)
                    embedded += 1
        return {"person_embeddings": embedded, "missing_person_embeddings": missing}


def build_reid_adapter(
    variant_name: str,
    config: Dict[str, Any],
    video_path: Optional[Path],
    backend_instance: Optional[Any] = None,
) -> Dict[str, Any]:
    """Build local OSNet adapter or safely skip unsupported SBS variants."""
    if variant_name == "botsort_style_no_reid_yolo11m":
        return {"status": "not_requested", "adapter": None}
    paths = config.get("paths", {})
    if "sbs" in variant_name:
        repo_path = Path(str(paths.get("botsort_repo_root", "")))
        weight_key = "botsort_sbs_mot17_weights" if "mot17" in variant_name else "botsort_sbs_mot20_weights"
        weights_path = Path(str(paths.get(weight_key, "")))
        missing = []
        if not repo_path.exists():
            missing.append("BoT-SORT repository")
        if not weights_path.exists():
            missing.append("SBS weights")
        if missing:
            reason = "missing optional dependency: %s" % ", ".join(missing)
        else:
            reason = "SBS/FastReID adapter is not enabled in this isolated internal benchmark"
        return {"status": "skipped", "reason": reason, "adapter": None}
    if video_path is None or not video_path.exists():
        return {"status": "skipped", "reason": "video_missing", "adapter": None}
    if variant_name == "botsort_osnet_finetuned_yolo11m":
        weights = Path(str(paths.get("osnet_finetuned_weights", "")))
        if not weights.exists():
            return {"status": "skipped", "reason": "fine-tuned OSNet weights missing: %s" % weights, "adapter": None}
        backend = backend_instance
        if backend is None:
            try:
                backend = FineTunedOSNetBackend(
                    weights,
                    device=str(config.get("osnet_finetuned", {}).get("device", "cuda")),
                )
            except Exception as exc:
                return {"status": "skipped", "reason": str(exc), "adapter": None}
    else:
        return {"status": "skipped", "reason": "unsupported ReID benchmark variant", "adapter": None}
    batch_size = int(config.get("osnet_finetuned", {}).get("batch_size", 128))
    return {
        "status": "ok",
        "backend": backend.name,
        "embedding_dim": backend.embedding_dim,
        "weights_loaded": backend.weights_loaded,
        "backend_instance": backend,
        "adapter": PersonReIDAdapter(backend, video_path, batch_size=batch_size),
    }


def _resolve_device(requested: str, torch_module: Any) -> str:
    if str(requested).startswith("cuda") and not torch_module.cuda.is_available():
        return "cpu"
    return str(requested)


def _strip_module_prefix(state: Dict[str, Any]) -> Dict[str, Any]:
    output = {}
    for key, value in state.items():
        text = str(key)
        if text.startswith("module."):
            text = text[len("module.") :]
        output[text] = value
    return output


def _compatible_state_dict(model: Any, state: Dict[str, Any]) -> Dict[str, Any]:
    model_state = model.state_dict()
    output = {}
    for key, value in state.items():
        if key not in model_state:
            continue
        try:
            if tuple(value.shape) != tuple(model_state[key].shape):
                continue
        except AttributeError:
            continue
        output[key] = value
    return output
