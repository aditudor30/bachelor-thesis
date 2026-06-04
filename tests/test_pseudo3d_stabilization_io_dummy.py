import numpy as np

from deep_oc_sort_3d.pseudo3d.pseudo3d_stabilization_io import (
    read_pseudo3d_outputs_csv,
    read_pseudo3d_outputs_jsonl,
    write_stabilized_outputs_csv,
    write_stabilized_outputs_jsonl,
)
from deep_oc_sort_3d.pseudo3d.pseudo3d_types import Pseudo3DOutput


def _output() -> Pseudo3DOutput:
    return Pseudo3DOutput(
        center_3d=np.asarray([1.0, 2.0, 3.0]),
        dimensions_3d=np.asarray([1.0, 2.0, 3.0]),
        yaw=0.0,
        depth=12.0,
        confidence_3d=0.7,
        center_3d_source="pseudo3d_bbox_height",
        dimensions_3d_source="class_prior",
        yaw_source="class_default",
        depth_source="bbox_height_prior",
        is_gt_derived=False,
        is_estimated_for_test=True,
        pseudo3d_method="bbox_height_depth",
        pseudo3d_version="0.2_stabilized",
        subset="test",
        split="test",
        scene_name="Warehouse_023",
        camera_id="Camera_0000",
        frame_id=1,
        class_id=0,
        class_name="Person",
        local_track_id=1,
        bbox_xyxy=(0.0, 0.0, 10.0, 20.0),
        confidence_2d=0.8,
        coordinate_frame="world",
    )


def test_read_write_stabilized_jsonl_csv(tmp_path) -> None:
    jsonl_path = tmp_path / "predictions.jsonl"
    csv_path = tmp_path / "predictions.csv"
    write_stabilized_outputs_jsonl([_output()], jsonl_path)
    write_stabilized_outputs_csv([_output()], csv_path)
    jsonl_rows = read_pseudo3d_outputs_jsonl(jsonl_path)
    csv_rows = read_pseudo3d_outputs_csv(csv_path)
    assert jsonl_rows[0].center_3d.tolist() == [1.0, 2.0, 3.0]
    assert csv_rows[0].dimensions_3d.tolist() == [1.0, 2.0, 3.0]
