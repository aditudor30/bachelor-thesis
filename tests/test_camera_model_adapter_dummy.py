import numpy as np

from deep_oc_sort_3d.pseudo3d.camera_model_adapter import backproject_pixel_with_depth, camera_model_from_calibration


def test_camera_model_adapter_extracts_intrinsics_and_backprojects():
    calibration = {"intrinsicMatrix": [[100.0, 0.0, 50.0], [0.0, 100.0, 50.0], [0.0, 0.0, 1.0]]}

    camera, error = camera_model_from_calibration(calibration)
    point, frame, point_error = backproject_pixel_with_depth(50.0, 50.0, 10.0, camera)

    assert error is None
    assert point_error is None
    assert frame == "camera"
    assert np.allclose(point, [0.0, 0.0, 10.0])


def test_camera_model_adapter_reports_missing_intrinsics():
    camera, error = camera_model_from_calibration({})

    assert camera is None
    assert error == "missing_intrinsics"

