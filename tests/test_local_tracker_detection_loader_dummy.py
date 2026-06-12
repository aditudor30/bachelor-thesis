from deep_oc_sort_3d.detection2d.yolo_detection_io import write_detections_csv
from deep_oc_sort_3d.detection2d.yolo_types import Detection2D
from deep_oc_sort_3d.local_tracker_benchmark.detection_loader import (
    group_detections_by_frame,
    inventory_detection_files,
    load_camera_detections,
)


def _detection(frame_id, confidence):
    return Detection2D(
        scene_id=20,
        scene_name="Warehouse_020",
        split="val",
        camera_id="Camera_0000",
        frame_id=frame_id,
        class_id=0,
        class_name="Person",
        confidence=confidence,
        bbox_xyxy=(10.0, 20.0, 30.0, 60.0),
        bbox_xywh=(10.0, 20.0, 20.0, 40.0),
        source="dummy",
    )


def test_detection_inventory_and_lazy_camera_load(tmp_path):
    pipeline_root = tmp_path / "pipeline"
    path = pipeline_root / "detections2d" / "official_val" / "Warehouse_020" / "Camera_0000.csv"
    write_detections_csv([_detection(0, 0.9), _detection(1, 0.01)], path)

    inventory, warnings = inventory_detection_files(
        pipeline_root,
        [("official_val", "val", "Warehouse_020")],
    )
    detections = load_camera_detections(inventory[0], min_confidence=0.05)
    grouped = group_detections_by_frame(detections)

    assert warnings == []
    assert len(inventory) == 1
    assert inventory[0]["camera_id"] == "Camera_0000"
    assert len(detections) == 1
    assert detections[0].frame_id == 0
    assert detections[0].matched_gt_object_id is None
    assert sorted(grouped.keys()) == [0]
