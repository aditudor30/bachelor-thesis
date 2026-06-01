import numpy as np

from deep_oc_sort_3d.observations.observation_io import write_observations_jsonl
from deep_oc_sort_3d.observations.observation_types import Observation3D
from deep_oc_sort_3d.pipeline.pipeline_summary import (
    aggregate_per_class_from_observations,
    write_inference_summary,
    write_per_class_summary,
)


def _observation(class_id, class_name, matched):
    return Observation3D(
        scene_id=20,
        scene_name="Warehouse_020",
        split="val",
        camera_id="Camera_0000",
        frame_id=0,
        detection_id=class_id,
        class_id=class_id,
        class_name=class_name,
        confidence=0.9,
        bbox_xyxy=(1.0, 2.0, 10.0, 20.0),
        bbox_xywh=(1.0, 2.0, 9.0, 18.0),
        center_3d=np.asarray([1.0, 2.0, 3.0], dtype=float),
        dimensions_3d=np.asarray([1.0, 1.0, 1.0], dtype=float),
        yaw=0.0,
        object_id=1 if matched else None,
        matched_gt=matched,
        matched_iou=0.8 if matched else None,
        depth_value=2.0,
        depth_sampling_method="center_median",
        source="yolo",
    )


def test_aggregate_per_class_from_observations(tmp_path):
    path = tmp_path / "observations3d" / "official_val" / "Warehouse_020" / "Camera_0000.jsonl"
    write_observations_jsonl([_observation(0, "Person", True), _observation(1, "Forklift", False)], path)

    summary = aggregate_per_class_from_observations([path])
    rows = summary["rows"]

    assert len(rows) == 2
    person = [row for row in rows if row["class_name"] == "Person"][0]
    assert person["subset"] == "official_val"
    assert person["matched_gt"] == 1
    assert person["mean_iou"] == 0.8


def test_summary_writers_create_csv_and_json(tmp_path):
    inference_path = tmp_path / "inference_summary.csv"
    class_csv = tmp_path / "per_class_summary.csv"
    class_json = tmp_path / "per_class_summary.json"

    write_inference_summary(
        [
            {
                "subset": "test",
                "split": "test",
                "scene_name": "Warehouse_023",
                "camera_id": "Camera_0000",
                "num_frames_processed": 5,
                "num_detections": 7,
                "detections_csv": "x.csv",
                "mot_like_path": "x.txt",
                "status": "ok",
                "error_message": "",
            }
        ],
        inference_path,
    )
    write_per_class_summary({"rows": []}, class_csv, class_json)

    assert inference_path.exists()
    assert class_csv.exists()
    assert class_json.exists()
