import numpy as np

from deep_oc_sort_3d.pseudo3d.pseudo3d_types import (
    Pseudo3DInput,
    Pseudo3DOutput,
    Pseudo3DPriors,
    pseudo3d_input_to_dict,
    pseudo3d_output_to_dict,
    pseudo3d_priors_to_dict,
)


def test_pseudo3d_types_serialize_to_plain_dicts():
    item = Pseudo3DInput(
        scene_name="Warehouse_023",
        camera_id="Camera_0000",
        frame_id=1,
        class_id=0,
        class_name="Person",
        bbox_xyxy=(1.0, 2.0, 3.0, 4.0),
        confidence=0.9,
        image_width=1920,
        image_height=1080,
        calibration={"cameraMatrix": [[1.0, 0.0, 0.0]]},
        track_id=10,
    )
    output = Pseudo3DOutput(
        center_3d=np.asarray([1.0, 2.0, 3.0]),
        dimensions_3d=np.asarray([0.7, 0.8, 1.7]),
        yaw=0.0,
        depth=5.0,
        confidence_3d=0.5,
        center_3d_source="pseudo3d_bbox_height",
        dimensions_3d_source="class_prior",
        yaw_source="motion_direction",
        depth_source="bbox_height_prior",
        is_gt_derived=False,
        is_estimated_for_test=True,
        pseudo3d_method="bbox_height_depth",
        pseudo3d_version="0.1",
    )
    prior = Pseudo3DPriors(0, "Person", 0.7, 0.8, 1.7, "high", "step15b")

    assert pseudo3d_input_to_dict(item)["bbox_xyxy"] == [1.0, 2.0, 3.0, 4.0]
    assert pseudo3d_output_to_dict(output)["center_3d"] == [1.0, 2.0, 3.0]
    assert pseudo3d_priors_to_dict(prior)["confidence_level"] == "high"

