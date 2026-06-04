import numpy as np

from deep_oc_sort_3d.pseudo3d.pseudo3d_io import (
    read_pseudo3d_predictions_csv,
    read_pseudo3d_predictions_jsonl,
    write_pseudo3d_predictions_csv,
    write_pseudo3d_predictions_jsonl,
)
from deep_oc_sort_3d.pseudo3d.pseudo3d_types import Pseudo3DOutput


def test_pseudo3d_io_writes_and_reads_predictions(tmp_path):
    output = Pseudo3DOutput(
        center_3d=np.asarray([1.0, 2.0, 3.0]),
        dimensions_3d=np.asarray([0.7, 0.8, 1.7]),
        yaw=0.0,
        depth=5.0,
        confidence_3d=0.5,
        center_3d_source="pseudo3d_bbox_height",
        dimensions_3d_source="class_prior",
        yaw_source="class_default",
        depth_source="bbox_height_prior",
        is_gt_derived=False,
        is_estimated_for_test=True,
        pseudo3d_method="bbox_height_depth",
        pseudo3d_version="0.1",
    )
    jsonl = tmp_path / "pred.jsonl"
    csv_path = tmp_path / "pred.csv"

    write_pseudo3d_predictions_jsonl([output], jsonl)
    write_pseudo3d_predictions_csv([output], csv_path)

    assert read_pseudo3d_predictions_jsonl(jsonl)[0]["center_3d"] == [1.0, 2.0, 3.0]
    assert read_pseudo3d_predictions_csv(csv_path)[0]["center_x"] == "1.0"

