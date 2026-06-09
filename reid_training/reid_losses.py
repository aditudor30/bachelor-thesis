"""Loss functions for OSNet Person ReID fine-tuning."""

from typing import Any, Dict, Tuple


def cross_entropy_loss(logits: Any, labels: Any) -> Any:
    """Cross-entropy loss wrapper."""
    import torch.nn.functional as F

    return F.cross_entropy(logits, labels)


def batch_hard_triplet_loss(features: Any, labels: Any, margin: float = 0.3) -> Any:
    """Compute batch-hard triplet loss on L2-normalized features."""
    import torch
    import torch.nn.functional as F

    if features.numel() == 0:
        return features.sum() * 0.0
    features = F.normalize(features, p=2, dim=1)
    distances = torch.cdist(features, features, p=2)
    labels_col = labels.view(-1, 1)
    positive_mask = labels_col.eq(labels_col.t())
    negative_mask = ~positive_mask
    eye = torch.eye(labels.shape[0], device=labels.device, dtype=torch.bool)
    positive_mask = positive_mask & (~eye)
    if positive_mask.sum().item() == 0 or negative_mask.sum().item() == 0:
        return features.sum() * 0.0
    hardest_positive = distances.masked_fill(~positive_mask, -1.0).max(dim=1)[0]
    hardest_negative = distances.masked_fill(~negative_mask, 1e6).min(dim=1)[0]
    valid = hardest_positive >= 0.0
    if valid.sum().item() == 0:
        return features.sum() * 0.0
    return F.relu(hardest_positive[valid] - hardest_negative[valid] + float(margin)).mean()


def total_reid_loss(outputs: Dict[str, Any], labels: Any, config: Dict[str, Any]) -> Tuple[Any, Dict[str, float]]:
    """Compute weighted CE + triplet loss and return loss plus detached components."""
    loss_cfg = config.get("loss", {})
    ce_weight = float(loss_cfg.get("ce_weight", 1.0))
    triplet_weight = float(loss_cfg.get("triplet_weight", 0.5))
    ce = cross_entropy_loss(outputs["logits"], labels)
    triplet = batch_hard_triplet_loss(outputs["features"], labels, margin=float(loss_cfg.get("triplet_margin", 0.3)))
    total = ce * ce_weight + triplet * triplet_weight
    components = {
        "total_loss": float(total.detach().cpu().item()),
        "ce_loss": float(ce.detach().cpu().item()),
        "triplet_loss": float(triplet.detach().cpu().item()),
    }
    return total, components
