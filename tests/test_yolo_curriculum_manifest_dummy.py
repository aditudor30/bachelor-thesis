import json

from deep_oc_sort_3d.detection2d.yolo_curriculum_manifest import (
    check_manifest_for_duplicates,
    read_curriculum_manifest,
    summarize_manifest,
    write_curriculum_manifest,
)


def _manifest_row(frame_id):
    return {
        "curriculum": "easy_allclass",
        "split": "train",
        "scene_name": "Warehouse_000",
        "scene_id": 0,
        "camera_id": "Camera_0000",
        "frame_id": frame_id,
        "image_path": "images/train/Warehouse_000_Camera_0000_%06d.jpg" % frame_id,
        "label_path": "labels/train/Warehouse_000_Camera_0000_%06d.txt" % frame_id,
        "class_counts_json": json.dumps({"Person": 1, "Forklift": 1}, sort_keys=True),
        "selected_target_classes_json": json.dumps(["Forklift"], sort_keys=True),
        "max_area_norm": 0.02,
        "mean_area_norm": 0.01,
        "difficulties_json": json.dumps({"easy": 2}, sort_keys=True),
        "score": 12.0,
        "source": "audit",
        "contains_person_only": False,
        "contains_rare_class": True,
    }


def test_curriculum_manifest_roundtrip_and_summary(tmp_path):
    path = tmp_path / "curriculum_manifest.csv"
    write_curriculum_manifest([_manifest_row(0)], path)

    rows = read_curriculum_manifest(path)
    summary = summarize_manifest(path)

    assert rows[0]["frame_id"] == 0
    assert rows[0]["class_counts"]["Forklift"] == 1
    assert rows[0]["selected_target_classes"] == ["Forklift"]
    assert summary["total_images"] == 1
    assert summary["per_class_counts"]["Person"] == 1
    assert summary["rare_class_frames"] == 1


def test_curriculum_manifest_duplicate_detection(tmp_path):
    path = tmp_path / "curriculum_manifest.csv"
    write_curriculum_manifest([_manifest_row(0), _manifest_row(0)], path)

    report = check_manifest_for_duplicates(path)

    assert report["num_rows"] == 2
    assert report["num_duplicates"] == 1
