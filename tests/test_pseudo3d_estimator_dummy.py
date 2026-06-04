from deep_oc_sort_3d.pseudo3d.pseudo3d_estimator import Pseudo3DEstimator
from deep_oc_sort_3d.pseudo3d.pseudo3d_priors import Pseudo3DPriorTable
from deep_oc_sort_3d.pseudo3d.pseudo3d_types import Pseudo3DInput, Pseudo3DPriors


def _config():
    return {
        "pseudo3d": {"version": "0.1", "require_world_coordinates": False},
        "method": {"primary": "bbox_height_depth"},
        "bbox_height_depth": {"min_bbox_height_px": 8, "projection_point": "bottom_center", "min_depth_m": 0.1, "max_depth_m": 100.0},
        "yaw": {"class_default_yaw": 0.0},
        "metadata": {"mark_estimated_for_test": True},
    }


def _input(bbox):
    return Pseudo3DInput(
        scene_name="Warehouse_020",
        camera_id="Camera_0000",
        frame_id=1,
        class_id=0,
        class_name="Person",
        bbox_xyxy=bbox,
        confidence=0.9,
        image_width=100,
        image_height=100,
        calibration={"intrinsicMatrix": [[100.0, 0.0, 50.0], [0.0, 100.0, 50.0], [0.0, 0.0, 1.0]]},
        subset="official_val",
        split="val",
    )


def test_pseudo3d_estimator_valid_input_has_class_prior_metadata():
    priors = Pseudo3DPriorTable({0: Pseudo3DPriors(0, "Person", 0.7, 0.8, 1.7, "high", "step15b")})
    estimator = Pseudo3DEstimator(priors, _config())

    output = estimator.estimate(_input((40.0, 20.0, 60.0, 70.0)))

    assert output.failure_reason is None
    assert output.dimensions_3d_source == "class_prior"
    assert output.center_3d_source == "pseudo3d_bbox_height"
    assert output.is_gt_derived is False
    assert output.is_estimated_for_test is True


def test_pseudo3d_estimator_invalid_input_reports_failure():
    priors = Pseudo3DPriorTable({0: Pseudo3DPriors(0, "Person", 0.7, 0.8, 1.7, "high", "step15b")})
    estimator = Pseudo3DEstimator(priors, _config())

    output = estimator.estimate(_input((40.0, 20.0, 60.0, 21.0)))

    assert output.failure_reason == "bbox_height_too_small"
    assert output.center_3d_source == "unknown"

