from deep_oc_sort_3d.detection2d.yolo_bbox_difficulty import (
    classify_bbox_difficulty,
    default_difficulty_config,
)


def test_classify_bbox_difficulty_defaults():
    assert classify_bbox_difficulty(100.0, 80.0, 0.02) == "easy"
    assert classify_bbox_difficulty(40.0, 30.0, 0.006) == "medium"
    assert classify_bbox_difficulty(20.0, 20.0, 0.002) == "hard"


def test_classify_bbox_difficulty_custom_thresholds():
    config = default_difficulty_config()
    config["easy_area_norm"] = 0.1
    config["medium_area_norm"] = 0.01
    config["medium_min_side"] = 10

    assert classify_bbox_difficulty(100.0, 100.0, 0.02, config) == "medium"

