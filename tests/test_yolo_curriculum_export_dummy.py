import csv

import numpy as np

from deep_oc_sort_3d.detection2d.yolo_curriculum_exporter import YoloCurriculumExporter
from deep_oc_sort_3d.detection2d.yolo_curriculum_manifest import read_curriculum_manifest
from deep_oc_sort_3d.detection2d.yolo_label_utils import read_yolo_label_file


def _write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _audit_row(frame_id, class_name, class_id, difficulty):
    return {
        "split": "train",
        "scene_name": "Warehouse_006",
        "scene_id": 6,
        "camera_id": "Camera_0011",
        "frame_id": frame_id,
        "object_id": frame_id + class_id + 10,
        "class_name": class_name,
        "class_id": class_id,
        "x1": 10,
        "y1": 10,
        "x2": 50,
        "y2": 50,
        "width_px": 40,
        "height_px": 40,
        "area_px": 1600,
        "aspect_ratio": 1.0,
        "image_width": 100,
        "image_height": 100,
        "width_norm": 0.4,
        "height_norm": 0.4,
        "area_norm": 0.16,
        "center_x_norm": 0.3,
        "center_y_norm": 0.3,
        "difficulty": difficulty,
    }


def test_yolo_curriculum_export_writes_dataset_manifest_and_labels(tmp_path, monkeypatch):
    scene = tmp_path / "train" / "Warehouse_006"
    (scene / "videos").mkdir(parents=True)
    (scene / "videos" / "Camera_0011.mp4").write_bytes(b"")
    audit_csv = tmp_path / "audit.csv"
    _write_csv(
        audit_csv,
        [
            _audit_row(0, "PalletTruck", 2, "easy"),
            _audit_row(0, "Person", 0, "hard"),
        ],
    )

    def fake_read_frame(_path, _frame_id):
        return np.zeros((100, 100, 3), dtype=np.uint8)

    monkeypatch.setattr(
        "deep_oc_sort_3d.detection2d.yolo_curriculum_exporter.safe_read_video_frame",
        fake_read_frame,
    )
    exporter = YoloCurriculumExporter(
        root=tmp_path,
        output_dir=tmp_path / "yolo_curriculum",
        audit_csv=audit_csv,
        curriculum="easy_allclass",
        target_classes=["PalletTruck", "Person"],
        allowed_difficulties=["easy"],
        max_frames_total=10,
        max_person_only_frames=0,
        min_area_norm=0.004,
        include_all_visible_objects=True,
    )

    records = exporter.export()
    manifest = tmp_path / "yolo_curriculum" / "curriculum_manifest.csv"
    data_yaml = tmp_path / "yolo_curriculum" / "data.yaml"
    labels = read_yolo_label_file(records[0].label_path)
    rows = read_curriculum_manifest(manifest)

    assert len(records) == 1
    assert data_yaml.exists()
    assert manifest.exists()
    assert len(labels) == 2
    assert rows[0]["class_counts"]["PalletTruck"] == 1
    assert rows[0]["class_counts"]["Person"] == 1
    assert rows[0]["contains_rare_class"] is True
