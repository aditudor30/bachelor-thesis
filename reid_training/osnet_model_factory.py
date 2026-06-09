"""OSNet model factory for local SmartSpaces fine-tuning."""

from pathlib import Path
from typing import Any, Dict, Tuple


def build_osnet_model(config: Dict[str, Any], num_classes: int, device: str) -> Any:
    """Build OSNet with a new classifier and local pretrained weights."""
    try:
        import torch
        import torchreid
    except ImportError as exc:
        raise ImportError("torch and torchreid are required for OSNet fine-tuning; no download is attempted.") from exc

    weights_path = Path(str(config.get("paths", {}).get("pretrained_osnet_weights", "")))
    if not weights_path.exists():
        raise FileNotFoundError("Local OSNet weights not found: %s" % weights_path)
    model_cfg = config.get("model", {})
    model = torchreid.models.build_model(
        name=str(model_cfg.get("architecture", "osnet_x1_0")),
        num_classes=int(num_classes),
        pretrained=False,
        loss="triplet",
    )
    state = torch.load(str(weights_path), map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    compatible = filter_compatible_state_dict(model, strip_module_prefix(state))
    model.load_state_dict(compatible, strict=False)
    model.to(device)
    return model


def resolve_device(config: Dict[str, Any]) -> str:
    """Resolve training device."""
    import torch

    requested = str(config.get("training", {}).get("device", "cuda"))
    if requested == "cuda":
        if torch.cuda.is_available():
            return "cuda"
        print("warning: CUDA requested but not available; falling back to CPU")
        return "cpu"
    return requested


def forward_with_features(model: Any, images: Any) -> Dict[str, Any]:
    """Forward OSNet and normalize output format to logits/features."""
    import torch.nn.functional as F

    output = model(images)
    if isinstance(output, tuple) or isinstance(output, list):
        logits = output[0]
        features = output[1] if len(output) > 1 else output[0]
    else:
        logits = output
        features = output
    if len(features.shape) > 2:
        features = features.reshape(features.shape[0], -1)
    return {"logits": logits, "features": F.normalize(features, p=2, dim=1)}


def extract_features(model: Any, images: Any) -> Any:
    """Extract L2-normalized features in eval mode."""
    import torch
    import torch.nn.functional as F

    was_training = bool(model.training)
    model.eval()
    with torch.no_grad():
        output = model(images)
        if isinstance(output, tuple) or isinstance(output, list):
            output = output[-1]
        if len(output.shape) > 2:
            output = output.reshape(output.shape[0], -1)
        features = F.normalize(output, p=2, dim=1)
    if was_training:
        model.train()
    return features


def strip_module_prefix(state: Any) -> Dict[str, Any]:
    """Strip DataParallel module prefix from checkpoint dict."""
    if not isinstance(state, dict):
        return {}
    output = {}
    for key, value in state.items():
        text = str(key)
        if text.startswith("module."):
            text = text[len("module.") :]
        output[text] = value
    return output


def filter_compatible_state_dict(model: Any, state: Dict[str, Any]) -> Dict[str, Any]:
    """Drop incompatible keys such as pretrained classifier weights."""
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


def set_backbone_trainable(model: Any, trainable: bool) -> None:
    """Freeze or unfreeze non-classifier parameters."""
    for name, param in model.named_parameters():
        if "classifier" in str(name):
            param.requires_grad = True
        else:
            param.requires_grad = bool(trainable)


def model_state_payload(model: Any, optimizer: Any, epoch: int, metrics: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    """Build checkpoint payload."""
    return {
        "epoch": int(epoch),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "metrics": metrics,
        "config": config,
    }

