import json

from deep_oc_sort_3d.detection2d.yolo_bbox_audit import audit_gt_bboxes, summarize_bbox_audit


def _write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def _create_scene(root):
    scene = root / "train" / "Warehouse_000"
    scene.mkdir(parents=True)
    _write_json(
        scene / "calibration.json",
        {
            "calibrationType": "dummy",
            "sensors": [
                {
                    "id": "Camera_0000",
                    "attributes": {
                        "frameWidth": 100,
                        "frameHeight": 100,
                    },
                }
            ],
        },
    )
    _write_json(
        scene / "ground_truth.json",
        {
            "0": [
                {
                    "object type": "Person",
                    "object id": 1,
                    "3d location": [0.0, 0.0, 0.0],
                    "3d bounding box scale": [1.0, 1.0, 1.0],
                    "3d bounding box rotation": [0.0, 0.0, 0.0],
                    "2d bounding box visible": {"Camera_0000": [10.0, 20.0, 30.0, 60.0]},
                },
                {
                    "object type": "Forklift",
                    "object id": 2,
                    "3d location": [0.0, 0.0, 0.0],
                    "3d bounding box scale": [1.0, 1.0, 1.0],
                    "3d bounding box rotation": [0.0, 0.0, 0.0],
                    "2d bounding box visible": {"Camera_0000": [40.0, 40.0, 90.0, 90.0]},
                },
            ]
        },
    )


def test_audit_gt_bboxes_creates_records(tmp_path):
    _create_scene(tmp_path)

    records = audit_gt_bboxes(
        root=tmp_path,
        split="train",
        scenes=["Warehouse_000"],
        camera_id="Camera_0000",
    )

    assert len(records) == 2
    person = [record for record in records if record.class_name == "Person"][0]
    assert person.width_px == 20.0
    assert person.height_px == 40.0
    assert person.area_px == 800.0
    assert person.aspect_ratio == 0.5
    assert person.area_norm == 0.08
    assert person.class_id == 0


def test_summarize_bbox_audit_reports_missing_classes(tmp_path):
    _create_scene(tmp_path)

    records = audit_gt_bboxes(tmp_path, "train", ["Warehouse_000"], camera_id="Camera_0000")
    summary = summarize_bbox_audit(records)

    assert summary["total_boxes"] == 2
    assert summary["count_per_class"]["Person"] == 1
    assert summary["count_per_class"]["Forklift"] == 1
    assert "missing class: PalletTruck" in summary["warnings"]

