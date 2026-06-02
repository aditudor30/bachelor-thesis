"""Dummy tests for final export validation."""

import numpy as np

from deep_oc_sort_3d.final_export.export_validation import validate_global_frame_records
from tests.test_generic_export_dummy import make_global_record


def test_validation_detects_invalid_bbox_duplicate_class_and_nan():
    a = make_global_record(0, 1)
    b = make_global_record(0, 1)
    c = make_global_record(1, 1)
    c.class_id = 2
    d = make_global_record(2, 2)
    d.bbox_xyxy = (10.0, 10.0, 5.0, 5.0)
    e = make_global_record(3, 3)
    e.center_3d = np.asarray([1.0, float("nan"), 3.0], dtype=float)
    report = validate_global_frame_records([a, b, c, d, e])
    assert report["num_errors"] >= 4
    assert any("duplicate_row" in item for item in report["errors"])
    assert any("global_track_class_inconsistency" in item for item in report["errors"])
    assert "invalid_bbox" in report["errors"]
    assert "center_3d_nan_or_inf" in report["errors"]
