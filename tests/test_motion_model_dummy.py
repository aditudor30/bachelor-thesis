import numpy as np

from deep_oc_sort_3d.tracking.motion_model import ConstantVelocity2D, ConstantVelocity3D, bbox_center_xyxy


def test_constant_velocity_3d_predicts_linear_motion():
    model = ConstantVelocity3D()
    model.update(np.asarray([0.0, 0.0, 0.0], dtype=float), frame_id=0)
    model.update(np.asarray([2.0, 0.0, 0.0], dtype=float), frame_id=2)

    predicted = model.predict(frame_id=3)

    np.testing.assert_allclose(predicted, np.asarray([3.0, 0.0, 0.0], dtype=float))


def test_constant_velocity_2d_predicts_bbox_center():
    model = ConstantVelocity2D()
    model.update((0.0, 0.0, 10.0, 10.0), frame_id=0)
    model.update((2.0, 0.0, 12.0, 10.0), frame_id=1)

    predicted = model.predict(frame_id=2)

    np.testing.assert_allclose(bbox_center_xyxy((0.0, 0.0, 10.0, 10.0)), np.asarray([5.0, 5.0]))
    np.testing.assert_allclose(predicted, np.asarray([9.0, 5.0], dtype=float))
