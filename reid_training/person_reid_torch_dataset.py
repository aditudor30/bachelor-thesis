"""PyTorch dataset for SmartSpaces Person ReID crops."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageEnhance

from deep_oc_sort_3d.reid_training.reid_dataset_io import read_csv_rows


def load_reid_rows(csv_path: Path, min_crops_per_identity: int = 5, max_crops_per_identity: Optional[int] = None) -> Tuple[List[Dict[str, Any]], Dict[str, int], List[str]]:
    """Load valid crop rows, filter rare identities, and map identities to labels."""
    rows, _fields = read_csv_rows(csv_path)
    valid = [row for row in rows if str(row.get("is_valid_crop", "1")) == "1"]
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for row in valid:
        groups.setdefault(str(row.get("identity_id", "")), []).append(row)
    kept_identities = sorted([identity for identity, values in groups.items() if len(values) >= int(min_crops_per_identity)])
    label_map = {identity: index for index, identity in enumerate(kept_identities)}
    excluded = sorted([identity for identity in groups.keys() if identity not in label_map])
    output: List[Dict[str, Any]] = []
    for identity in kept_identities:
        values = sorted(groups[identity], key=lambda row: (str(row.get("camera_id", "")), int(row.get("frame_id", -1))))
        if max_crops_per_identity is not None and len(values) > int(max_crops_per_identity):
            values = _uniform_sample(values, int(max_crops_per_identity))
        for row in values:
            item = dict(row)
            item["label"] = int(label_map[identity])
            output.append(item)
    return output, label_map, excluded


class SmartSpacesPersonReIDTorchDataset(object):
    """Lazy crop dataset for ReID fine-tuning/evaluation."""

    def __init__(
        self,
        csv_path: Path,
        min_crops_per_identity: int = 5,
        max_crops_per_identity: Optional[int] = None,
        input_size: Tuple[int, int] = (256, 128),
        training: bool = True,
        normalize: bool = True,
    ) -> None:
        self.csv_path = Path(csv_path)
        self.rows, self.identity_to_label, self.excluded_identities = load_reid_rows(
            self.csv_path,
            min_crops_per_identity=min_crops_per_identity,
            max_crops_per_identity=max_crops_per_identity,
        )
        self.input_size = input_size
        self.training = bool(training)
        self.normalize = bool(normalize)

    def __len__(self) -> int:
        """Return number of crop rows."""
        return len(self.rows)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        """Read crop lazily and return tensors/metadata."""
        import torch

        row = self.rows[index]
        image = read_crop_image(Path(str(row.get("crop_path", ""))))
        image = transform_image(
            image,
            input_size=self.input_size,
            training=self.training,
            normalize=self.normalize,
        )
        return {
            "image": torch.from_numpy(image).float(),
            "label": torch.tensor(int(row.get("label", -1)), dtype=torch.long),
            "identity_id": str(row.get("identity_id", "")),
            "crop_path": str(row.get("crop_path", "")),
            "scene_name": str(row.get("scene_name", "")),
            "camera_id": str(row.get("camera_id", "")),
            "frame_id": int(row.get("frame_id", -1)),
            "object_id": int(row.get("object_id", -1)),
        }


def read_crop_image(path: Path) -> Image.Image:
    """Read RGB crop image."""
    if not path.exists():
        raise FileNotFoundError("Missing ReID crop: %s" % path)
    return Image.open(str(path)).convert("RGB")


def transform_image(image: Image.Image, input_size: Tuple[int, int], training: bool = True, normalize: bool = True) -> np.ndarray:
    """Resize and convert image to CHW float tensor array."""
    height, width = int(input_size[0]), int(input_size[1])
    image = image.resize((width, height), Image.BILINEAR)
    if training:
        image = _augment_pil(image)
    arr = np.asarray(image, dtype=np.float32) / 255.0
    if normalize:
        mean = np.asarray([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
        std = np.asarray([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)
        arr = (arr - mean) / std
    return arr.transpose(2, 0, 1)


def _augment_pil(image: Image.Image) -> Image.Image:
    """Apply lightweight deterministic-free PIL augmentations using numpy RNG."""
    if np.random.rand() < 0.5:
        image = image.transpose(Image.FLIP_LEFT_RIGHT)
    if np.random.rand() < 0.35:
        factor = float(np.random.uniform(0.85, 1.15))
        image = ImageEnhance.Color(image).enhance(factor)
    if np.random.rand() < 0.35:
        factor = float(np.random.uniform(0.85, 1.15))
        image = ImageEnhance.Brightness(image).enhance(factor)
    return image


def _uniform_sample(rows: List[Dict[str, Any]], count: int) -> List[Dict[str, Any]]:
    if len(rows) <= int(count):
        return list(rows)
    indices = np.linspace(0, len(rows) - 1, int(count))
    return [rows[int(round(index))] for index in indices]

