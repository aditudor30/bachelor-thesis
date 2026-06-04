import numpy as np

from deep_oc_sort_3d.pseudo3d.pseudo3d_stabilizer import Pseudo3DStabilizer
from deep_oc_sort_3d.pseudo3d.pseudo3d_types import Pseudo3DOutput


def _output(frame_id: int, x_value: float) -> Pseudo3DOutput:
    return Pseudo3DOutput(
        center_3d=np.asarray([x_value, 0.0, 0.0]),
        dimensions_3d=np.asarray([1.0, 2.0, 3.0]),
        yaw=0.0,
        depth=10.0,
        confidence_3d=0.8,
        center_3d_source="pseudo3d_bbox_height",
        dimensions_3d_source="class_prior",
        yaw_source="class_default",
        depth_source="bbox_height_prior",
        is_gt_derived=False,
        is_estimated_for_test=True,
        pseudo3d_method="bbox_height_depth",
        pseudo3d_version="0.1",
        subset="official_val",
        split="val",
        scene_name="Warehouse_020",
        camera_id="Camera_0000",
        frame_id=frame_id,
        class_id=0,
        class_name="Person",
        local_track_id=7,
        bbox_xyxy=(0.0, 0.0, 20.0, 20.0),
        confidence_2d=0.9,
        coordinate_frame="world",
    )


def test_stabilizer_changes_jump_metadata_and_keeps_dimensions() -> None:
    config = {
        "center_smoothing": {"enabled": False},
        "depth_smoothing": {"enabled": False},
        "jump_guard": {"enabled": True, "max_step_m": 6.0, "strategy": "hold_previous", "reduce_confidence_factor": 0.5},
        "small_bbox_guard": {"enabled": False},
        "yaw": {"recompute_from_smoothed_motion": False},
        "metadata": {"pseudo3d_version": "0.2_stabilized"},
    }
    outputs, report = Pseudo3DStabilizer(config).stabilize_track([_output(1, 0.0), _output(2, 20.0)])
    assert report["num_jump_corrected"] == 1
    assert outputs[1].center_3d_source == "pseudo3d_jump_guarded"
    assert outputs[1].confidence_3d < 0.8
    assert outputs[1].dimensions_3d_source == "class_prior"
    assert outputs[1].dimensions_3d.tolist() == [1.0, 2.0, 3.0]
