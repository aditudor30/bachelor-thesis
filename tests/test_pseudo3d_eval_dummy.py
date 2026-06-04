from deep_oc_sort_3d.pseudo3d.pseudo3d_eval import build_eval_record


def test_pseudo3d_eval_computes_center_and_dimension_error():
    prediction = {
        "subset": "official_val",
        "scene_name": "Warehouse_020",
        "camera_id": "Camera_0000",
        "frame_id": 1,
        "class_id": 0,
        "class_name": "Person",
        "center_x": 1.0,
        "center_y": 2.0,
        "center_z": 3.0,
        "width_3d": 1.0,
        "length_3d": 2.0,
        "height_3d": 3.0,
        "yaw": 0.0,
    }
    gt = {"center_3d": [1.0, 2.0, 5.0], "dimensions_3d": [1.0, 2.0, 4.0], "yaw": 0.0}

    record = build_eval_record(prediction, gt)

    assert record.center_error == 2.0
    assert record.depth_error == 2.0
    assert record.dimensions_error == 1.0

