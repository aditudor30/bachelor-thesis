import csv

import numpy as np

from deep_oc_sort_3d.visualization3d.visualization_io import (
    filter_records_by_class,
    filter_records_by_frame,
    filter_records_by_global_track_id,
    load_global_frame_records_csv,
    parse_center_dimensions_yaw_from_record,
)


def test_visualization_io_filters_and_parses(tmp_path):
    path = tmp_path / "records.csv"
    fieldnames = [
        "frame_id",
        "global_track_id",
        "class_name",
        "center_x",
        "center_y",
        "center_z",
        "width_3d",
        "length_3d",
        "height_3d",
        "yaw",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(
            {
                "frame_id": "10",
                "global_track_id": "7",
                "class_name": "Person",
                "center_x": "1.0",
                "center_y": "2.0",
                "center_z": "3.0",
                "width_3d": "0.5",
                "length_3d": "0.6",
                "height_3d": "1.7",
                "yaw": "0.1",
            }
        )
    records = load_global_frame_records_csv(path)
    assert len(filter_records_by_frame(records, 10)) == 1
    assert len(filter_records_by_global_track_id(records, 7)) == 1
    assert len(filter_records_by_class(records, "Person")) == 1
    parsed = parse_center_dimensions_yaw_from_record(records[0])
    assert parsed is not None
    center, dimensions, yaw = parsed
    np.testing.assert_allclose(center, np.asarray([1.0, 2.0, 3.0]))
    np.testing.assert_allclose(dimensions, np.asarray([0.5, 0.6, 1.7]))
    assert yaw == 0.1
