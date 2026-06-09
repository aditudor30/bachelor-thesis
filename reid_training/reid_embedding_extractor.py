"""Batch-wise embedding extraction for OSNet ReID models."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from deep_oc_sort_3d.reid_training.osnet_model_factory import extract_features
from deep_oc_sort_3d.reid_training.person_reid_torch_dataset import SmartSpacesPersonReIDTorchDataset
from deep_oc_sort_3d.reid_training.reid_dataset_io import progress_iter, write_csv_rows


EMBEDDING_METADATA_FIELDS = [
    "index",
    "identity_id",
    "label",
    "crop_path",
    "scene_name",
    "camera_id",
    "frame_id",
    "object_id",
]


def extract_embeddings_for_csv(
    model: Any,
    csv_path: Path,
    output_npy: Path,
    output_metadata_csv: Path,
    config: Dict[str, Any],
    device: str,
    max_crops: Optional[int] = None,
    show_progress: bool = True,
) -> Dict[str, Any]:
    """Extract embeddings for crops from a metadata CSV."""
    import torch
    from torch.utils.data import DataLoader, Subset

    data_cfg = config.get("data", {})
    model_cfg = config.get("model", {})
    dataset = SmartSpacesPersonReIDTorchDataset(
        csv_path=csv_path,
        min_crops_per_identity=max(2, int(data_cfg.get("min_crops_per_identity", 5))),
        max_crops_per_identity=None,
        input_size=(int(model_cfg.get("input_height", 256)), int(model_cfg.get("input_width", 128))),
        training=False,
    )
    if max_crops is not None and len(dataset) > int(max_crops):
        indices = np.linspace(0, len(dataset) - 1, int(max_crops))
        dataset_for_loader = Subset(dataset, [int(round(index)) for index in indices])
    else:
        dataset_for_loader = dataset
    loader = DataLoader(
        dataset_for_loader,
        batch_size=int(data_cfg.get("batch_size", 64)),
        shuffle=False,
        num_workers=int(data_cfg.get("num_workers", 0)),
    )
    embeddings: List[np.ndarray] = []
    metadata: List[Dict[str, Any]] = []
    model.eval()
    with torch.no_grad():
        for batch_index, batch in enumerate(progress_iter(loader, show_progress, "extract ReID embeddings", "batch")):
            images = batch["image"].to(device)
            features = extract_features(model, images).detach().cpu().numpy()
            embeddings.append(features)
            batch_size = features.shape[0]
            for item_index in range(batch_size):
                metadata.append(
                    {
                        "index": len(metadata),
                        "identity_id": str(batch["identity_id"][item_index]),
                        "label": int(batch["label"][item_index]),
                        "crop_path": str(batch["crop_path"][item_index]),
                        "scene_name": str(batch["scene_name"][item_index]),
                        "camera_id": str(batch["camera_id"][item_index]),
                        "frame_id": int(batch["frame_id"][item_index]),
                        "object_id": int(batch["object_id"][item_index]),
                    }
                )
    matrix = np.vstack(embeddings) if embeddings else np.zeros((0, int(model_cfg.get("embedding_dim", 512))), dtype=np.float32)
    output_npy.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(output_npy), matrix.astype(np.float32))
    write_csv_rows(metadata, output_metadata_csv, EMBEDDING_METADATA_FIELDS)
    return {"embeddings": str(output_npy), "metadata": str(output_metadata_csv), "num_embeddings": int(matrix.shape[0]), "embedding_dim": int(matrix.shape[1]) if matrix.ndim == 2 else 0}


def load_embedding_matrix(npy_path: Path, metadata_csv: Path) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    """Load embeddings and metadata."""
    from deep_oc_sort_3d.reid_training.reid_dataset_io import read_csv_rows

    matrix = np.load(str(npy_path))
    rows, _fields = read_csv_rows(metadata_csv)
    return matrix, rows

