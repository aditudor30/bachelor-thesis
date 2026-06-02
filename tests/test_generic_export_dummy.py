"""Dummy tests for generic final export."""

import numpy as np

from deep_oc_sort_3d.final_export.generic_export import (
    GENERIC_EXPORT_FIELDS,
    export_generic_tracking_scene_csv,
    read_global_frame_records_csv,
    write_global_frame_records_csv,
)
from deep_oc_sort_3d.final_export.global_frame_types import GlobalFrameRecord


def make_global_record(frame_id, global_track_id):
    return GlobalFrameRecord(
        scene_id=20,
        scene_name="Warehouse_020",
        split="val",
        subset="official_val",
        camera_id="Camera_0000",
        frame_id=frame_id,
        global_track_id=global_track_id,
        local_track_id=1,
        candidate_id="c1",
        detection_id=frame_id,
        class_id=0,
        class_name="Person",
        confidence=0.9,
        bbox_xyxy=(1.0, 2.0, 11.0, 22.0),
        bbox_xywh=(1.0, 2.0, 10.0, 20.0),
        center_3d=np.asarray([1.0, 2.0, 3.0], dtype=float),
        dimensions_3d=np.asarray([0.5, 1.0, 1.8], dtype=float),
        yaw=0.1,
        matched_gt_object_id=5,
        matched_gt=True,
        source="dummy",
    )


def test_generic_export_writes_sorted_columns(tmp_path):
    records_path = tmp_path / "Camera_0000_global_records.csv"
    output_path = tmp_path / "Warehouse_020.csv"
    write_global_frame_records_csv([make_global_record(2, 9), make_global_record(1, 8)], records_path)
    summary = export_generic_tracking_scene_csv([records_path], output_path, drop_unassigned=True)
    rows = output_path.read_text(encoding="utf-8").splitlines()
    assert rows[0].split(",") == GENERIC_EXPORT_FIELDS
    assert summary["rows_written"] == 2
    assert read_global_frame_records_csv(records_path)[0].frame_id == 2
