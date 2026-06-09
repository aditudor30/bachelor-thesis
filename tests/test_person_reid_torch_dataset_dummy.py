import csv

import numpy as np
from PIL import Image

from deep_oc_sort_3d.reid_training.person_reid_torch_dataset import SmartSpacesPersonReIDTorchDataset, load_reid_rows


def _write_crop(path, color):
    image = Image.fromarray(np.full((32, 16, 3), color, dtype=np.uint8))
    image.save(str(path))


def test_person_reid_torch_dataset_loads_dummy_crops(tmp_path):
    crop_a = tmp_path / "a.jpg"
    crop_b = tmp_path / "b.jpg"
    _write_crop(crop_a, 80)
    _write_crop(crop_b, 160)
    csv_path = tmp_path / "metadata.csv"
    fields = ["identity_id", "crop_path", "is_valid_crop", "scene_name", "camera_id", "frame_id", "object_id"]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerow({"identity_id": "id_1", "crop_path": str(crop_a), "is_valid_crop": "1", "scene_name": "Warehouse_000", "camera_id": "Camera_0000", "frame_id": "0", "object_id": "10"})
        writer.writerow({"identity_id": "id_1", "crop_path": str(crop_b), "is_valid_crop": "1", "scene_name": "Warehouse_000", "camera_id": "Camera_0000", "frame_id": "1", "object_id": "10"})
    rows, label_map, excluded = load_reid_rows(csv_path, min_crops_per_identity=2)
    assert len(rows) == 2
    assert label_map == {"id_1": 0}
    assert excluded == []
    dataset = SmartSpacesPersonReIDTorchDataset(csv_path, min_crops_per_identity=2, input_size=(64, 32), training=False)
    sample = dataset[0]
    assert tuple(sample["image"].shape) == (3, 64, 32)
    assert int(sample["label"]) == 0
    assert sample["identity_id"] == "id_1"
